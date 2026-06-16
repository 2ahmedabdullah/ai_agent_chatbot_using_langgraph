"""Conditional edges for the public-agent graph."""

from __future__ import annotations

from typing_extensions import Literal

from graph.states.public_agent_state import PublicAgentState


def route_after_abuse(state: PublicAgentState) -> Literal["blocked", "continue"]:
    """Stop if abuse filter blocked the query."""
    if state.get("route") == "blocked" or state.get("is_abusive"):
        return "blocked"
    return "continue"


def route_after_vague(state: PublicAgentState) -> Literal["clarify", "continue"]:
    """Stop for clarification if query is vague."""
    if state.get("route") == "clarify" or state.get("is_vague"):
        return "clarify"
    return "continue"


def route_after_cache(state: PublicAgentState) -> Literal["cache_hit", "cache_miss"]:
    """End early if FAQ cache hit."""
    if state.get("cache_hit") or state.get("route") == "cached_answer":
        return "cache_hit"
    return "cache_miss"


def route_after_quality(state: PublicAgentState) -> Literal["accepted", "retry", "fallback_human"]:
    """Route after quality gate."""
    if state.get("route") == "fallback_human":
        return "fallback_human"
    if state.get("route") == "retry":
        return "retry"
    decision = state.get("quality_decision") or {}
    status = decision.get("status")
    if status == "fallback_human":
        return "fallback_human"
    if status == "retry":
        return "retry"
    return "accepted"
