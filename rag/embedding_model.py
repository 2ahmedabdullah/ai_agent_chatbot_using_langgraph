"""
LangChain embedding manager for document chunks and user queries.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Own the LangChain OpenAIEmbeddings instance used by RAG."""

    MODEL_NAME = "text-embedding-3-small"
    DIMENSION = 1536

    def __init__(self) -> None:
        self.model_name = self.MODEL_NAME
        self.dimension = self.DIMENSION
        self.api_key = os.getenv("PUBLIC_AGENT_OPENAI_APIKEY")

        if not self.api_key:
            raise ValueError("PUBLIC_AGENT_OPENAI_APIKEY not set")

        self.embeddings = OpenAIEmbeddings(
            model=self.model_name,
            api_key=self.api_key,
            max_retries=3,
            timeout=60,
        )
        # logger.info("[EMBEDDINGS] Ready: %s (dim=%s)", self.model_name, self.dimension)

    def embed_documents(self, documents: Iterable[Document | str]) -> List[List[float]]:
        texts = [self._text_from_item(item) for item in documents]
        texts = [text for text in texts if text.strip()]
        if not texts:
            return []
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        return self.embeddings.embed_query(query.strip())

    @property
    def model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "provider": "openai",
        }

    @staticmethod
    def _text_from_item(item: Document | str) -> str:
        if isinstance(item, Document):
            return item.page_content
        if isinstance(item, str):
            return item
        raise TypeError(f"Unsupported embedding input: {type(item)!r}")
