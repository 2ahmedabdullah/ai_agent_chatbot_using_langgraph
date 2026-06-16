from __future__ import annotations

from types import SimpleNamespace

from agents.abuse_handler import BLOCKED_RESPONSE, abuse_filter_node
from agents.greeting_handler import greeting_handler_node
from agents.vague_handler import CLARIFICATION_RESPONSE, vague_handler_node


def test_greeting_handler_node_returns_greeting_response(monkeypatch, base_state):
    responses = [
        SimpleNamespace(content="YES"),
        SimpleNamespace(content="Hello! I am Mesa Public Agent. How can I help you today?"),
    ]

    def fake_invoke_llm_with_backoff(_messages):
        return responses.pop(0)

    monkeypatch.setattr("agents.greeting_handler.invoke_llm_with_backoff", fake_invoke_llm_with_backoff)
    result = greeting_handler_node({**base_state, "user_query": "Hi"})

    assert isinstance(result.get("final_response"), str)
    assert result["final_response"]
    assert result["route"] == "greeting"


def test_vague_handler_node_returns_clarification_format(monkeypatch, base_state):
    responses = [
        SimpleNamespace(content="VAGUE"),
        SimpleNamespace(content="Could you please share more details so I can assist?"),
    ]

    def fake_invoke_llm_with_backoff(_messages):
        return responses.pop(0)

    monkeypatch.setattr("agents.vague_handler.invoke_llm_with_backoff", fake_invoke_llm_with_backoff)
    result = vague_handler_node({**base_state, "user_query": "Explain"})

    assert isinstance(result.get("final_response"), str)
    assert result["final_response"]
    assert result["is_vague"] is True
    assert result["route"] == "clarify"


def test_abuse_filter_node_blocks_toxic_content(monkeypatch, base_state):
    def fake_invoke_llm_with_backoff(_messages):
        return SimpleNamespace(content="BLOCKED")

    monkeypatch.setattr("agents.abuse_handler.invoke_llm_with_backoff", fake_invoke_llm_with_backoff)
    result = abuse_filter_node({**base_state, "user_query": "You are trash"})

    assert result["is_abusive"] is True
    assert result["route"] == "blocked"
    assert isinstance(result.get("final_response"), str)
    assert result["final_response"] == BLOCKED_RESPONSE


def test_vague_handler_fallback_response_on_llm_error(monkeypatch, base_state):
    def fake_invoke_llm_with_backoff(_messages):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr("agents.vague_handler.invoke_llm_with_backoff", fake_invoke_llm_with_backoff)
    result = vague_handler_node({**base_state, "user_query": "Okay"})

    assert result["is_vague"] is True
    assert result["route"] == "clarify"
    assert result["final_response"] == CLARIFICATION_RESPONSE
