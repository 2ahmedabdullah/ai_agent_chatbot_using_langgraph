from __future__ import annotations

from types import SimpleNamespace

from graph.builder import router_node


def test_router_node_classifies_toxic(monkeypatch, base_state):
    expected = SimpleNamespace(classification="toxic", reasoning="contains harassment")

    def fake_invoke_llm_with_backoff(_structured_llm, _messages):
        return expected

    monkeypatch.setattr("graph.agents.utils.invoke_llm_with_backoff", fake_invoke_llm_with_backoff)
    result = router_node({**base_state, "user_query": "You are useless."})

    assert result["classification"] == "toxic"
    assert "router_reasoning" in result["metadata"]


def test_router_node_classifies_greeting(monkeypatch, base_state):
    expected = SimpleNamespace(classification="greeting", reasoning="simple salutation")

    def fake_invoke_llm_with_backoff(_structured_llm, _messages):
        return expected

    monkeypatch.setattr("graph.agents.utils.invoke_llm_with_backoff", fake_invoke_llm_with_backoff)
    result = router_node({**base_state, "user_query": "Hi there"})

    assert result["classification"] == "greeting"
    assert result["metadata"]["router_reasoning"] == "simple salutation"


def test_router_node_classifies_vague(monkeypatch, base_state):
    expected = SimpleNamespace(classification="vague", reasoning="too short and ambiguous")

    def fake_invoke_llm_with_backoff(_structured_llm, _messages):
        return expected

    monkeypatch.setattr("graph.agents.utils.invoke_llm_with_backoff", fake_invoke_llm_with_backoff)
    result = router_node({**base_state, "user_query": "Help?"})

    assert result["classification"] == "vague"
    assert "ambiguous" in result["metadata"]["router_reasoning"]


def test_router_node_classifies_valid(monkeypatch, base_state):
    expected = SimpleNamespace(classification="valid", reasoning="public-service information request")

    def fake_invoke_llm_with_backoff(_structured_llm, _messages):
        return expected

    monkeypatch.setattr("graph.agents.utils.invoke_llm_with_backoff", fake_invoke_llm_with_backoff)
    result = router_node(
        {
            **base_state,
            "user_query": "How do I apply for a trade license in Mesa?",
        }
    )

    assert result["classification"] == "valid"
    assert "router_reasoning" in result["metadata"]
