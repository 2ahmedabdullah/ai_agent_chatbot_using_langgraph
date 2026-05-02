"""Vague-query handling for the public-agent graph."""
import os
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.agents.utils import invoke_llm_with_backoff

# Constants
CLARIFICATION_RESPONSE = "I’m happy to help. Could you share a little more detail?"

# Initialize LLM with your specific key
# Using gpt-4o-mini for speed and cost efficiency
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0, # Set to 0 for consistent classification
    api_key=os.getenv("PUBLIC_AGENT_OPENAI_APIKEY")
)

def vague_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure LLM-based vague query detection.
    Determines if a query is too ambiguous to answer and asks for specifics.
    """
    query = (state.get("user_query", "") or "").strip()
    
    # If the query is completely empty, mark as vague immediately
    if not query:
        state["is_vague"] = True
        state["final_response"] = CLARIFICATION_RESPONSE
        state["route"] = "clarify"
        return state

    try:
        # Step 1: Analyze if the query is actionable
        decision_prompt = (
            "Analyze if the user query is too vague to provide a helpful response. "
            "Examples of vague: 'Help', 'Why?', 'Explain', 'Okay'. "
            "Examples of actionable: 'How do I login?', 'What is this chatbot?'. "
            "Reply ONLY with 'VAGUE' or 'ACTIONABLE'."
        )
        
        decision = invoke_llm_with_backoff([
            SystemMessage(content=decision_prompt),
            HumanMessage(content=query)
        ])

        if "VAGUE" in decision.content.upper():
            # Step 2: Generate a polite, specific request for more info
            clarification = invoke_llm_with_backoff([
                SystemMessage(content=(
                    "The user's query is too vague. Ask them politely to provide more context "
                    "or specifics so you can assist them better."
                )),
                HumanMessage(content=query)
            ])
            
            state["is_vague"] = True
            state["final_response"] = clarification.content
            state["route"] = "clarify"

    except Exception as e:
        print(f"Vague Handler LLM Error: {e}")
        # If API fails completely, use the default clarification text
        state["is_vague"] = True
        state["final_response"] = CLARIFICATION_RESPONSE
        state["route"] = "clarify"
                
    return state