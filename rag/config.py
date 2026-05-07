"""Shared RAG configuration.

Keep indexing and retrieval on the same Qdrant collection. The public-agent
RAG pipeline only stores chunked documents here; FAQ cache and chat history
stay in MongoDB under the caching package.
"""

from __future__ import annotations

import os


DEFAULT_RAG_COLLECTION_NAME = "documents"


def get_rag_collection_name() -> str:
    """Return the Qdrant collection used by both indexing and retrieval."""
    configured_name = os.getenv("RAG_QDRANT_COLLECTION", DEFAULT_RAG_COLLECTION_NAME).strip()
    return configured_name or DEFAULT_RAG_COLLECTION_NAME
