"""
Cold cache for public-agent FAQ/query-response reuse.

This is the first cache lookup after the abuse handler. It checks:
1. exact normalized query match
2. semantic query match using text-embedding-3-small and cosine similarity

It stores only query/response pairs, not document chunks. Document chunks remain
inside the RAG vector store.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.collection import Collection
except ImportError:  # pragma: no cover - reported clearly at runtime
    ASCENDING = DESCENDING = MongoClient = Collection = None

logger = logging.getLogger(__name__)

MONGO_URI_ENV = "CHATBOT_RW_CHAT_HISTORY_URL"
DB_NAME_ENV = "MONGODB_DATABASE_NAME"
OPENAI_API_KEY_ENV = "PUBLIC_AGENT_OPENAI_APIKEY"

DEFAULT_DB_NAME = "chatbot_db"
DEFAULT_COLLECTION_NAME = "public_agent_cache_messages"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TTL_DAYS = 10
DEFAULT_MAX_CACHE_ITEMS = 20
DEFAULT_SEMANTIC_THRESHOLD = 0.80


def _utcnow() -> datetime:
    return datetime.utcnow()


def normalize(text: str) -> str:
    """Normalize query text for exact cache matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity without requiring numpy."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b) #test numpy speed latency


def _require_pymongo() -> None:
    if MongoClient is None:
        raise RuntimeError("pymongo is required for cold cache. Install pymongo in the project environment.")


def get_cache_collection(collection_name: str = DEFAULT_COLLECTION_NAME) -> Collection:
    """Return the Mongo collection used for global FAQ cache entries."""
    _require_pymongo()

    mongo_uri = os.getenv(MONGO_URI_ENV)
    if not mongo_uri:
        raise ValueError(f"{MONGO_URI_ENV} not set")

    db_name = os.getenv(DB_NAME_ENV, DEFAULT_DB_NAME)
    mongo_client = MongoClient(mongo_uri)
    collection = mongo_client[db_name][collection_name]

    collection.create_index([("user_query", ASCENDING)], unique=True)
    collection.create_index([("last_accessed", ASCENDING)])
    collection.create_index([("hit_count", ASCENDING)])
    return collection


def get_openai_client() -> OpenAI:
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{OPENAI_API_KEY_ENV} not set")
    return OpenAI(api_key=api_key)


class SemanticFAQCache:
    """Mongo-backed exact and semantic cache for generated public-agent answers."""

    def __init__(
        self,
        collection: Optional[Collection] = None,
        client: Optional[OpenAI] = None,
        ttl_days: int = DEFAULT_TTL_DAYS,
        max_items: int = DEFAULT_MAX_CACHE_ITEMS,
        threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ) -> None:
        self.collection = collection or get_cache_collection()
        self.client = client or get_openai_client()
        self.ttl_days = ttl_days
        self.max_items = max_items
        self.threshold = threshold

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding

    def get_cache_result(self, user_input: str) -> Optional[str]:
        """Return cached response if exact or semantic cache hit exists."""
        normalized_query = normalize(user_input)
        if not normalized_query:
            return None

        self.prune_expired()

        exact_hit = self.collection.find_one({"user_query": normalized_query})
        if exact_hit:
            self._touch(exact_hit["_id"])
            # logger.info("[FAQ_CACHE] Exact cache hit")
            return exact_hit.get("response")

        # logger.info("[FAQ_CACHE] Exact miss; checking semantic cache")
        query_vector = self.get_embedding(normalized_query)
        best_doc: Optional[Dict[str, Any]] = None
        best_score = -1.0

        for cached_doc in self.collection.find({"embedding": {"$exists": True}}):
            score = cosine_similarity(query_vector, cached_doc["embedding"])
            if score > best_score:
                best_score = score
                best_doc = cached_doc

        # logger.info("[FAQ_CACHE] Best semantic score: %.4f", best_score)
        if best_doc and best_score >= self.threshold:
            self._touch(best_doc["_id"], semantic_score=best_score)
            logger.info("[FAQ_CACHE] Semantic cache hit")
            return best_doc.get("response")

        return None

    def set_cache_result(self, user_query: str, response: str) -> None:
        """Insert or update one normalized query/response pair."""
        normalized_query = normalize(user_query)
        if not normalized_query or not response or not response.strip():
            return

        self.prune_expired()
        embedding = self.get_embedding(normalized_query)
        now = _utcnow()

        self.collection.update_one(
            {"user_query": normalized_query},
            {
                "$set": {
                    "response": response,
                    "embedding": embedding,
                    "last_accessed": now,
                    "expires_at": now + timedelta(days=self.ttl_days),
                    "embedding_model": EMBEDDING_MODEL,
                },
                "$setOnInsert": {"created_at": now},
                "$inc": {"hit_count": 1},
            },
            upsert=True,
        )
        self.enforce_max_items()

    def prune_expired(self) -> int:
        cutoff = _utcnow() - timedelta(days=self.ttl_days)
        result = self.collection.delete_many(
            {
                "$or": [
                    {"last_accessed": {"$lt": cutoff}},
                    {"expires_at": {"$lte": _utcnow()}},
                ]
            }
        )
        return int(result.deleted_count)

    def enforce_max_items(self) -> int:
        """Keep the cache bounded by deleting the coldest entries."""
        overflow = self.collection.count_documents({}) - self.max_items
        if overflow <= 0:
            return 0

        victims = list(
            self.collection.find({})
            .sort([("last_accessed", ASCENDING), ("hit_count", ASCENDING)])
            .limit(overflow)
        )
        victim_ids = [victim["_id"] for victim in victims]
        if not victim_ids:
            return 0

        result = self.collection.delete_many({"_id": {"$in": victim_ids}})
        return int(result.deleted_count)

    def clear_cache(self) -> int:
        result = self.collection.delete_many({})
        return int(result.deleted_count)

    def list_cache(self, limit: int = DEFAULT_MAX_CACHE_ITEMS) -> List[Dict[str, Any]]:
        rows = self.collection.find({}, {"embedding": 0}).sort("last_accessed", DESCENDING).limit(limit)
        return [self._public_cache_item(row) for row in rows] 

    def check_cache_node(self, user_input: str) -> Dict[str, Any]:
        """Graph-node friendly cache lookup result."""
        response = self.get_cache_result(user_input)
        return {
            "cache_hit": response is not None,
            "cached_response": response,
        }

    def _touch(self, document_id: Any, semantic_score: Optional[float] = None) -> None:
        update: Dict[str, Any] = {
            "last_accessed": _utcnow(),
            "expires_at": _utcnow() + timedelta(days=self.ttl_days),
        }
        if semantic_score is not None:
            update["last_semantic_score"] = semantic_score

        self.collection.update_one(
            {"_id": document_id},
            {
                "$set": update,
                "$inc": {"hit_count": 1},
            },
        )

    @staticmethod
    def _public_cache_item(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "user_query": row.get("user_query"),
            "response": row.get("response"),
            "hit_count": row.get("hit_count", 0),
            "created_at": row.get("created_at"),
            "last_accessed": row.get("last_accessed"),
            "expires_at": row.get("expires_at"),
            "embedding_model": row.get("embedding_model"),
            "last_semantic_score": row.get("last_semantic_score"),
        }


def get_cache_result(user_input: str, threshold: float = DEFAULT_SEMANTIC_THRESHOLD) -> Optional[str]:
    return SemanticFAQCache(threshold=threshold).get_cache_result(user_input)


def check_cache(user_input: str, threshold: float = DEFAULT_SEMANTIC_THRESHOLD) -> Dict[str, Any]:
    """Convenience function for the public-agent check_cache graph node."""
    return SemanticFAQCache(threshold=threshold).check_cache_node(user_input)


def set_cache_result(user_query: str, response: str) -> None:
    SemanticFAQCache().set_cache_result(user_query, response)


def clear_cache() -> int:
    return SemanticFAQCache().clear_cache()


def list_cache(limit: int = DEFAULT_MAX_CACHE_ITEMS) -> List[Dict[str, Any]]:
    return SemanticFAQCache().list_cache(limit=limit)
