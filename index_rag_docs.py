"""Index rag/rag_docs into Qdrant for the public-agent chatbot."""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from rag.config import get_rag_collection_name
from rag.data_ingestion import load_rag_docs
from rag.retriever import RAGDocumentRetriever


def index_rag_docs() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    chunks, report = load_rag_docs()
    print(f"Loaded {len(chunks)} chunk(s) from {len(report)} file(s).")

    retriever = RAGDocumentRetriever()
    print(f"Using Qdrant collection: {get_rag_collection_name()}")
    indexed_count = retriever.add_documents(chunks)
    print(f"Indexed {indexed_count} chunk(s) into Qdrant.")

    failed = [item for item in report if item.get("error")]
    if failed:
        print("Some files failed during ingestion:")
        for item in failed:
            print(f"- {item.get('filename')}: {item.get('error')}")


if __name__ == "__main__":
    index_rag_docs()
