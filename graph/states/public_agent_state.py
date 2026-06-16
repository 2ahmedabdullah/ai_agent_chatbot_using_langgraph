from __future__ import annotations
from typing import Any, Dict, List, Optional
from typing_extensions import Literal, TypedDict

# ... RouteName, RouteDecision, and QualityDecision remain the same ...

class RouteDecision(TypedDict, total=False):
    """LLM router output."""
 
    route: Literal["rag", "clarify", "fallback_human"]
    intent: str
    reason: str
    confidence: float
 
 
class QualityDecision(TypedDict, total=False):
    """Final-answer quality gate output."""
 
    status: Literal["accepted", "retry", "fallback_human"]
    reason: str
    confidence: float
 

class PublicAgentState(TypedDict, total=False): # Added total=False
    """Shared graph state passed between public-agent nodes."""

    user_query: str
    session_id: Optional[str]
    user_id: Optional[str]

    sanitized_query: str
    route: RouteDecision 

    is_abusive: bool
    abuse_reasons: List[str]
    is_vague: bool
    vague_reason: Optional[str]

    chat_history: List[Dict[str, Any]]
    cache_hit: bool
    cached_response: Optional[str]

    retrieved_context: List[Dict[str, Any]]
    raw_response: Optional[str]
    final_response: Optional[str]
    quality_decision: QualityDecision

    retry_count: int
    max_retries: int
    hot_memory_limit: int
    rag_top_k: int
    cache_final_answer: bool

    metadata: Dict[str, Any]
    
    classification: str

    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost: float


