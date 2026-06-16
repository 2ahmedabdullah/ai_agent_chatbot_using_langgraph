from __future__ import annotations

from graph.builder import build_graph


def test_valid_query_end_to_end_graph_invoke(monkeypatch, base_state):
    def fake_check_cache_node(_state):
        return {"cache_hit": False}

    def fake_router_node(_state):
        return {"classification": "valid", "metadata": {"router_reasoning": "valid civic query"}}

    def fake_rag_node(state):
        return {
            "route": "rag_answer",
            "retrieved_context": [{"content": "Mesa office hours are 9 to 5.", "metadata": {}, "score": 0.9}],
            "raw_response": "Mesa offices typically operate from 9 AM to 5 PM.",
            "metadata": {**dict(state.get("metadata") or {}), "rag": {"context_count": 1}},
        }

    def fake_quality_check_node(_state):
        return {
            "quality_decision": {
                "status": "accepted",
                "reason": "response grounded in retrieved context",
                "confidence": 0.98,
            }
        }

    def fake_store_cache_node(_state):
        return {"metadata": {"cache_store": "ok"}}

    def fake_finalize_response_node(state):
        text = state.get("raw_response") or ""
        return {
            "final_response": text,
            "metadata": {**dict(state.get("metadata") or {}), "finalized": True},
        }

    def fake_persist_memory_node(_state):
        return {"metadata": {"persisted": True}}

    monkeypatch.setattr("graph.builder.check_cache_node", fake_check_cache_node)
    monkeypatch.setattr("graph.builder.router_node", fake_router_node)
    monkeypatch.setattr("graph.builder.rag_node", fake_rag_node)
    monkeypatch.setattr("graph.builder.quality_check_node", fake_quality_check_node)
    monkeypatch.setattr("graph.builder.store_cache_node", fake_store_cache_node)
    monkeypatch.setattr("graph.builder.finalize_response_node", fake_finalize_response_node)
    monkeypatch.setattr("graph.builder.persist_memory_node", fake_persist_memory_node)

    graph = build_graph()
    result = graph.invoke(base_state)

    assert "final_response" in result
    assert isinstance(result["final_response"], str)
    assert result["final_response"]
    assert "metadata" in result
    assert isinstance(result["metadata"], dict)
