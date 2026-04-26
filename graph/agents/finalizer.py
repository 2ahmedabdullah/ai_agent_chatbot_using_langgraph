"""Final response shaping and quality nodes."""

from __future__ import annotations

from graph.agents.utils import merge_metadata
from graph.states.public_agent_state import PublicAgentState
from graph.tools.llm_tools import check_response_quality, finalize_response


def finalize_response_node(state: PublicAgentState) -> dict:
    """Supervisor final pass over raw node output."""
    raw_response = state.get("raw_response") or state.get("final_response") or ""
    try:
        final = finalize_response(
            query=state.get("user_query", ""),
            raw_response=raw_response,
            route=state.get("route", "fallback_human"),
        )
        return {"final_response": final}
    except Exception as exc:
        return {
            "final_response": raw_response,
            "metadata": merge_metadata(state, finalize_error=str(exc)),
        }


def quality_check_node(state: PublicAgentState) -> dict:
    """Quality gate after final response generation."""
    route = state.get("route", "fallback_human")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)

    if route in {"blocked", "clarify", "cached_answer"}:
        return {
            "quality_decision": {
                "status": "accepted",
                "reason": "terminal guardrail or cache route",
                "confidence": 1.0,
            }
        }

    try:
        decision = check_response_quality(
            query=state.get("user_query", ""),
            final_response=state.get("final_response") or "",
            route=route,
        )
    except Exception as exc:
        return {
            "quality_decision": {
                "status": "accepted",
                "reason": "quality check unavailable",
                "confidence": 0.0,
            },
            "metadata": merge_metadata(state, quality_error=str(exc)),
        }

    if decision.get("status") == "retry" and retry_count < max_retries:
        return {
            "route": "retry",
            "retry_count": retry_count + 1,
            "quality_decision": decision,
        }

    if decision.get("status") == "fallback_human":
        return {
            "route": "fallback_human",
            "quality_decision": decision,
            "final_response": "I am not confident enough to answer that safely. A human should review this.",
        }

    return {
        "quality_decision": decision,
    }
