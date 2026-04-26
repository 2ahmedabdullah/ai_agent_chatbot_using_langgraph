"""Shared graph-node helpers."""

from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4
from graph.states.public_agent_state import PublicAgentState

from tenacity import retry, stop_after_attempt, wait_exponential
import logging

def ensure_session_id(state: Dict[str, Any]) -> str:
    """Get or create a session id."""
    return state.get("session_id") or str(uuid4())


def merge_metadata(state: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
    """Return merged metadata without mutating caller-owned dictionaries."""
    metadata = dict(state.get("metadata") or {})
    metadata.update(updates)
    return metadata

def update_usage_metadata(state: PublicAgentState, response_metadata: dict) -> dict:
    """Extracts tokens and calculates estimated cost."""
    token_usage = response_metadata.get("token_usage", {})
    prompt = token_usage.get("prompt_tokens", 0)
    completion = token_usage.get("completion_tokens", 0)
    
    # Simple estimate for gpt-4o-mini
    # (Prices change, but tracking the count is the main goal)
    return {
        "prompt_tokens": state.get("prompt_tokens", 0) + prompt,
        "completion_tokens": state.get("completion_tokens", 0) + completion,
        "total_tokens": state.get("total_tokens", 0) + prompt + completion
    }

logger = logging.getLogger("public-agent")

# Universal retry logic: 
# Retries 3 times, waiting 2s, then 4s, then 8s between attempts.
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def invoke_llm_with_backoff(llm, messages):
    return llm.invoke(messages)