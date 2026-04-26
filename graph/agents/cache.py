"""FAQ cache nodes."""

from __future__ import annotations

from graph.agents.utils import merge_metadata
from graph.prompts.persona import FALLBACK_MESSAGE
from graph.states.public_agent_state import PublicAgentState
from graph.tools.cache_tools import lookup_faq_cache, store_faq_cache


def check_cache_node(state: PublicAgentState) -> dict:
    """Check exact/semantic FAQ cache."""
    try:
        result = lookup_faq_cache(state.get("user_query", ""))
    except Exception as exc:
        return {
            "cache_hit": False,
            "cached_response": None,
            "metadata": merge_metadata(state, cache_error=str(exc)),
        }

    if result.get("cache_hit"):
        response = result.get("cached_response")
        return {
            "route": "cached_answer",
            "cache_hit": True,
            "cached_response": response,
            "raw_response": response,
            "final_response": response,
        }

    return {
        "route": "continue",
        "cache_hit": False,
        "cached_response": None,
    }


def store_cache_node(state: PublicAgentState) -> dict:
    """Store successful non-cached final answers in FAQ cache."""
    route = state.get("route")
    if route != "rag_answer":
        return {}
    if state.get("cache_final_answer") is False:
        return {}
    if (state.get("final_response") or "").strip() == FALLBACK_MESSAGE:
        return {}

    try:
        store_faq_cache(state.get("user_query", ""), state.get("final_response") or "")
    except Exception as exc:
        return {"metadata": merge_metadata(state, cache_store_error=str(exc))}
    return {}
