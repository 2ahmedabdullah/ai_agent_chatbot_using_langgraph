"""Vague-query node."""

from __future__ import annotations

from agents.vague_handler import check_vague_query

from graph.states.public_agent_state import PublicAgentState


def vague_handler(state: PublicAgentState) -> dict:
    """Ask for clarification when the query is too vague."""
    result = check_vague_query(state.get("user_query", ""))
    updates = {
        "is_vague": result.is_vague,
        "vague_reason": result.reason,
    }
    if result.is_vague:
        updates.update(
            {
                "route": "clarify",
                "raw_response": result.response,
                "final_response": result.response,
            }
        )
    else:
        updates["route"] = "continue"
    return updates
