"""Main-graph RAG node."""

from __future__ import annotations

from graph.agents.utils import merge_metadata
from graph.states.public_agent_state import PublicAgentState
from rag.subgraph import invoke_rag

from graph.agents.utils import update_usage_metadata

def rag_node(state: PublicAgentState) -> dict:
    """Run the RAG subgraph as one main-graph node."""
    try:
        result = invoke_rag(
            query=state.get("user_query", ""),
            chat_history=state.get("chat_history", []),
            top_k=state.get("rag_top_k", 5),
        )
        # We are adding usage tracking here
        usage = update_usage_metadata(state, result.get("metadata", {}))
        
        return {
            "route": "rag_answer",
            "retrieved_context": result.get("retrieved_context", []),
            "raw_response": result.get("raw_response"),
            "metadata": merge_metadata(state, rag=result.get("metadata", {})),
            # This line saves the token counts to your state
            **usage 
        }
    except Exception as exc:
        fallback = (
            "I sincerely apologize, but I don't have the information needed to answer your request at this time. "
            "We are currently working hard to expand my capabilities and knowledge base to serve you better. "
            "For now, please reach out to **support@konguess.com**, and our team will be happy to assist you "
            "manually with this feature or data point."
        )
        return {
            "route": "fallback_human",
            "raw_response": fallback,
            "final_response": fallback,
            "metadata": merge_metadata(state, generation_error=str(exc)),
        }
