"""Hot-memory nodes."""

from __future__ import annotations

from graph.agents.utils import merge_metadata
from graph.states.public_agent_state import PublicAgentState
from graph.tools.memory_tools import load_hot_memory, store_turn


def load_hot_memory_node(state: PublicAgentState) -> dict:
    """Load recent chat history into graph state."""
    try:
        history = load_hot_memory(
            session_id=state.get("session_id") or "",
            user_id=state.get("user_id"),
            limit=state.get("hot_memory_limit", 2),
        )
        return {"chat_history": history}
    except Exception as exc:
        return {
            "chat_history": [],
            "metadata": merge_metadata(state, hot_memory_error=str(exc)),
        }


def persist_memory_node(state: PublicAgentState) -> dict:
    """Store final response in cache/hot memory where appropriate."""
    metadata = dict(state.get("metadata") or {})
    query = state.get("user_query", "")
    final_response = state.get("final_response") or ""

    if not final_response:
        return {}

    try:
        store_turn(
            session_id=state.get("session_id") or "",
            query=query,
            response=final_response,
            user_id=state.get("user_id"),
        )
    except Exception as exc:
        metadata["hot_memory_store_error"] = str(exc)

    return {"metadata": metadata}
