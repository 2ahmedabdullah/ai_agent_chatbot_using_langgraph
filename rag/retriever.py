from __future__ import annotations

"""
LangChain retrieval and vector indexing for RAG document chunks.

This file is only for chunked document knowledge. FAQ caching and chat history
live in the caching package because they run before classify_query in the
public-agent graph.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

try:
    from .config import get_rag_collection_name
    from .embedding_model import EmbeddingManager
except ImportError:
    from config import get_rag_collection_name
    from embedding_model import EmbeddingManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RAGDocumentRetriever:
    """Index and retrieve LangChain Documents from Qdrant."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ) -> None:

        self.collection_name = collection_name or get_rag_collection_name()
        self.embedding_manager = embedding_manager or EmbeddingManager()

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url:
            raise ValueError("QDRANT_URL is not set")

        # logger.info(f"[RAG] Connecting to Qdrant at: {self.qdrant_url}")

        # --- CONNECT ---
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=120,
            check_compatibility=False,
        )

        # --- VERIFY CONNECTION ---
        try:
            collections = self.client.get_collections()
            logger.info("[RAG] Qdrant connection successful")
            logger.info("[RAG] Existing collections: %s", [c.name for c in collections.collections])
        except Exception as e:
            logger.error("[RAG] Qdrant connection FAILED: %s", str(e))
            raise RuntimeError(
                "Qdrant is not reachable. Check URL/API key or start local instance."
            )

        # --- ENSURE COLLECTION ---
        self.ensure_collection()

        # --- VECTOR STORE ---
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_manager.embeddings,
        )

    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create collection if it does not exist."""
        collections = self.client.get_collections().collections
        names = {collection.name for collection in collections}

        if self.collection_name in names:
            logger.info("[RAG] Using existing collection: %s", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_manager.dimension,
                distance=Distance.COSINE,
            ),
        )

        logger.info("[RAG] Created collection: %s", self.collection_name)

    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> int:
        """Insert documents into Qdrant."""
        if not documents:
            logger.warning("[RAG] No documents to add")
            return 0

        ids = [self._point_id(document) for document in documents]

        self.vectorstore.add_documents(documents=documents, ids=ids)

        logger.info("[RAG] Indexed %d document chunks", len(documents))
        return len(documents)

    # ------------------------------------------------------------------

    def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores."""

        threshold = self._score_threshold(score_threshold)

        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=top_k,
            )
        except Exception as e:
            logger.error("[RAG] Retrieval failed: %s", str(e))
            return []

        logger.info("[RAG] Raw retrieved docs: %d", len(results))

        filtered = [(doc, score) for doc, score in results if score >= threshold]

        logger.info(
            "[RAG] After threshold filter: %d (threshold=%.2f)",
            len(filtered),
            threshold,
        )

        return filtered

    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return retrieval results in graph-friendly format."""
        docs = self.retrieve_documents(query, top_k, score_threshold)

        return [
            {
                "content": document.page_content,
                "metadata": document.metadata,
                "score": score,
            }
            for document, score in docs
        ]

    # ------------------------------------------------------------------

    @staticmethod
    def _point_id(document: Document) -> str:
        metadata = document.metadata

        raw = "|".join(
            [
                str(metadata.get("source", "")),
                str(metadata.get("page", "")),
                str(metadata.get("global_chunk_index", "")),
                document.page_content[:100],
            ]
        )

        return str(uuid5(NAMESPACE_URL, raw))

    # ------------------------------------------------------------------

    @staticmethod
    def _score_threshold(score_threshold: Optional[float]) -> float:
        if score_threshold is not None:
            return score_threshold
        return float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))


# Backward compatibility
ColdMemoryRetriever = RAGDocumentRetriever