import os
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.agents.utils import invoke_llm_with_backoff

# Constants
GREETING_RESPONSE = "Hi, I’m glad you’re here. What would you like to know?"

# Initialize LLM 
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.7, 
    api_key=os.getenv("PUBLIC_AGENT_OPENAI_APIKEY")
)

def greeting_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure LLM-based greeting detection and response.
    """
    query = (state.get("user_query", "") or "").strip()
    
    if not query:
        return state

    try:
        # Step 1: Classify if it is a greeting
        classification_response = invoke_llm_with_backoff([
            SystemMessage(content="You are a classifier. Is the user's input primarily a greeting, introduction, or small talk? Reply ONLY with 'YES' or 'NO'."),
            HumanMessage(content=query)
        ])
        
        is_greeting = "YES" in classification_response.content.upper()

        if is_greeting:
            # Step 2: Generate a natural response
            # You can either use your constant GREETING_RESPONSE or let the LLM be creative
            generation_response = invoke_llm_with_backoff([
                SystemMessage(content="The user just greeted you. Provide a warm, brief response saying you are Mesa Public Agent and ask how you can help them today."),
                HumanMessage(content=query)
            ])
            
            state["final_response"] = generation_response.content
            state["route"] = "greeting"

    except Exception as e:
        print(f"Greeting Handler LLM Error: {e}")
        # If the API fails completely, use the static greeting you defined at the top
        state["final_response"] = GREETING_RESPONSE
        state["route"] = "greeting"
        
    return state