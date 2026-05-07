"""
LangChain document chunking for RAG cold memory.

This file is intentionally a building block, not a graph. The public agent's
main LangGraph can call this as part of its RAG node or indexing job.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Split loaded LangChain Documents into retrieval-sized chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Return LangChain Documents with chunk metadata attached."""
        source_documents = list(documents)
        # logger.info("[CHUNKING] Chunking %s document(s)", len(source_documents))

        chunks = self.splitter.split_documents(source_documents)
        if not chunks:
            logger.warning("[CHUNKING] No chunks created")
            return []

        total_chunks = len(chunks)
        per_source_counts: dict[str, int] = {}

        for chunk in chunks:
            source = str(chunk.metadata.get("source", "unknown"))
            per_source_counts[source] = per_source_counts.get(source, 0) + 1
            chunk.metadata["chunk_index"] = per_source_counts[source] - 1
            chunk.metadata["memory_type"] = "cold"

        for global_index, chunk in enumerate(chunks):
            chunk.metadata["global_chunk_index"] = global_index
            chunk.metadata["total_chunks"] = total_chunks

        # logger.info("[CHUNKING] Created %s chunk(s)", total_chunks)
        return chunks
