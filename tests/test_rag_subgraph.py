from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

from rag.retriever import RAGDocumentRetriever
from rag.subgraph import generate_answer_node, retrieve_context_node


@pytest.mark.live_rag
def test_retrieve_context_node_returns_grounded_context():
    """
    Live retrieval integration test:
    Uses local Qdrant collection already indexed by index_rag_docs.py.
    """
    preflight = RAGDocumentRetriever().retrieve("leadership ai future of leadership", top_k=3)
    assert preflight, "Preflight retrieval returned no data from configured Qdrant collection."

    state = {"query": "leadership ai future of leadership", "top_k": 3, "metadata": {}}
    result = retrieve_context_node(state)

    retrieved = result.get("retrieved_context") or []
    assert retrieved, "Expected non-empty retrieval from local Qdrant."
    assert result["metadata"]["retrieved_context_count"] > 0

    first = retrieved[0]
    assert "content" in first
    assert isinstance(first["content"], str)
    assert first["content"].strip()
    assert "metadata" in first


@pytest.mark.live_rag
def test_generate_answer_node_returns_live_grounded_answer():
    """
    Live generation integration test:
    Retrieves context from local Qdrant, then generates an answer with real model call.
    """
    retrieval_state = {"query": "ethics ai talent evaluation", "top_k": 3, "metadata": {}}
    retrieval_result = retrieve_context_node(retrieval_state)
    retrieved = retrieval_result.get("retrieved_context") or []
    if not retrieved:
        pytest.fail("Cannot test generation because retrieval returned no documents.")

    state = {
        "query": retrieval_state["query"],
        "retrieved_context": retrieved,
        "metadata": {"test_case": "rag_generate_live"},
    }
    result = generate_answer_node(state)

    assert "raw_response" in result
    assert isinstance(result["raw_response"], str)
    assert result["raw_response"].strip()
    assert result["metadata"]["context_count"] >= 1
    assert result["metadata"]["test_case"] == "rag_generate_live"
