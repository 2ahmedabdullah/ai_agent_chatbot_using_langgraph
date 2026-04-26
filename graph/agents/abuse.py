"""Abuse-filter node."""

from __future__ import annotations

from agents.abuse_handler import check_abuse

from graph.agents.utils import ensure_session_id, merge_metadata
from graph.states.public_agent_state import PublicAgentState


def abuse_filter(state: PublicAgentState) -> dict:
    """Sanitize and block abusive/prompt-injection inputs."""
    result = check_abuse(state.get("user_query", ""))
    updates = {
        "session_id": ensure_session_id(state),
        "sanitized_query": result.sanitized_query,
        "user_query": result.sanitized_query,
        "is_abusive": result.is_blocked,
        "abuse_reasons": result.reasons,
        "metadata": merge_metadata(state, abuse_filter=result.metadata),
    }
    if result.is_blocked:
        updates.update(
            {
                "route": "blocked",
                "raw_response": result.response,
                "final_response": result.response,
            }
        )
    else:
        updates["route"] = "continue"
    return updates
