"""
Hot memory for the public-agent graph.

Hot memory is recent chat history. The cache node can load this immediately
after abuse filtering and place it into graph state before classification,
retrieval, or generation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
from typing import Any, Dict, List, Optional

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.collection import Collection
except ImportError:  # pragma: no cover - reported clearly at runtime
    ASCENDING = DESCENDING = MongoClient = Collection = None

logger = logging.getLogger(__name__)

MONGO_URI_ENV = "CHATBOT_RW_CHAT_HISTORY_URL"
DB_NAME_ENV = "MONGODB_DATABASE_NAME"
DEFAULT_DB_NAME = "chatbot_db"
DEFAULT_COLLECTION_NAME = "hot_memory_messages"
DEFAULT_TTL_SECONDS = 60 * 60
DEFAULT_HISTORY_LIMIT = 2


def _utcnow() -> datetime:
    return datetime.utcnow()


def _require_pymongo() -> None:
    if MongoClient is None:
        raise RuntimeError("pymongo is required for hot memory. Install pymongo in the project environment.")


def get_hot_memory_collection(collection_name: str = DEFAULT_COLLECTION_NAME) -> Collection:
    """Return the Mongo collection used for recent chat history."""
    _require_pymongo()

    mongo_uri = os.getenv(MONGO_URI_ENV)
    if not mongo_uri:
        raise ValueError(f"{MONGO_URI_ENV} not set")

    db_name = os.getenv(DB_NAME_ENV, DEFAULT_DB_NAME)
    mongo_client = MongoClient(mongo_uri)
    collection = mongo_client[db_name][collection_name]

    collection.create_index([("session_id", ASCENDING), ("created_at", DESCENDING)])
    collection.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    collection.create_index("expires_at", expireAfterSeconds=0)
    return collection


class HotMemoryStore:
    """Mongo-backed recent chat history store."""

    def __init__(
        self,
        collection: Optional[Collection] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> None:
        self.collection = collection or get_hot_memory_collection()
        self.ttl_seconds = ttl_seconds
        self.history_limit = history_limit

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store one chat message in hot memory."""
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be user, assistant, or system")
        if not session_id:
            raise ValueError("session_id is required")
        if not content or not content.strip():
            raise ValueError("content is required")

        now = _utcnow()
        document = {
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content.strip(),
            "metadata": metadata or {},
            "created_at": now,
            "last_accessed": now,
            "expires_at": now + timedelta(seconds=self.ttl_seconds),
        }
        result = self.collection.insert_one(document)
        return str(result.inserted_id)

    def get_history(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load recent chat history for the current graph state."""
        if not session_id:
            raise ValueError("session_id is required")

        query: Dict[str, Any] = {
            "session_id": session_id,
            "expires_at": {"$gt": _utcnow()},
        }
        if user_id:
            query["user_id"] = user_id

        max_items = limit or self.history_limit
        rows = list(
            self.collection.find(query)
            .sort("created_at", DESCENDING)
            .limit(max_items)
        )
        rows.reverse()

        ids = [row["_id"] for row in rows]
        if ids:
            self.collection.update_many(
                {"_id": {"$in": ids}},
                {"$set": {"last_accessed": _utcnow()}},
            )

        return [self._public_message(row) for row in rows]

    def clear_history(self, session_id: str, user_id: Optional[str] = None) -> int:
        """Delete hot memory for one session."""
        query: Dict[str, Any] = {"session_id": session_id}
        if user_id:
            query["user_id"] = user_id
        result = self.collection.delete_many(query)
        return int(result.deleted_count)

    def prune_expired(self) -> int:
        """Manual prune for environments where Mongo TTL has not run yet."""
        result = self.collection.delete_many({"expires_at": {"$lte": _utcnow()}})
        return int(result.deleted_count)

    @staticmethod
    def _public_message(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": row.get("role", ""),
            "content": row.get("content", ""),
            "timestamp": row.get("created_at"),
            "metadata": row.get("metadata", {}),
        }


def load_user_chat_history(
    session_id: str,
    user_id: Optional[str] = None,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> List[Dict[str, Any]]:
    """Convenience function for the public-agent cache node."""
    return HotMemoryStore().get_history(session_id=session_id, user_id=user_id, limit=limit)


def add_chat_message(
    session_id: str,
    role: str,
    content: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience function for appending a graph turn to hot memory."""
    return HotMemoryStore().add_message(
        session_id=session_id,
        role=role,
        content=content,
        user_id=user_id,
        metadata=metadata,
    )
