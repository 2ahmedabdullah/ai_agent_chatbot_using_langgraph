from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from aws_secrets import load_aws_secrets

# Load env vars early
load_dotenv()
load_aws_secrets()  # ← Load secrets from AWS Secrets Manager and inject into environment

# Import your graph components
from graph.builder import graph, stream_graph_events

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("public-agent-prod")

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform startup tasks here (e.g., connect to MongoDB or Qdrant)
    logger.info("Starting up Public Agent Chatbot API...")
    yield
    # Perform cleanup tasks here
    logger.info("Shutting down...")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Public Agent Chatbot",
    version="1.0.0",
    description="Production API for Public Agent",
    lifespan=lifespan,
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod, replace with actual frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schema Definitions ---
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    hot_memory_limit: int = Field(default=2, ge=0, le=5)
    rag_top_k: int = Field(default=5, ge=1, le=10)
    cache_final_answer: bool = True

class ChatResponse(BaseModel):
    answer: str
    session_id: Optional[str]
    route: Optional[str]
    cache_hit: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- Internal Helpers ---
def _request_to_state(request: ChatRequest) -> Dict[str, Any]:
    return {
        "user_query": request.query,
        "session_id": request.session_id,
        "user_id": request.user_id,
        "hot_memory_limit": request.hot_memory_limit,
        "rag_top_k": request.rag_top_k,
        "cache_final_answer": request.cache_final_answer,
        "retry_count": 0,
        "max_retries": 1,
        "metadata": {},
    }

def _state_to_response(state: Dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        answer=state.get("final_response") or state.get("raw_response") or "I'm sorry, I couldn't generate a response.",
        session_id=state.get("session_id"),
        route=state.get("route"),
        cache_hit=bool(state.get("cache_hit", False)),
        metadata=state.get("metadata") or {},
    )

# --- Endpoints ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """Liveness check for container orchestrators."""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Standard non-streaming chat endpoint."""
    try:
        # invoke is a blocking call, but FastAPI runs it in a threadpool
        result = graph.invoke(_request_to_state(request))
        return _state_to_response(result)
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

import uuid  # Ensure this is at the top of your file with other imports

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    active_session_id = request.session_id
    if not active_session_id or active_session_id == "string":
        active_session_id = str(uuid.uuid4())[:8]

    async def event_generator():
        try:
            state = _request_to_state(request)
            state["session_id"] = active_session_id 

            for event in stream_graph_events(state):
                # 1. Identify which node just finished
                node_name = list(event.keys())[0]
                node_data = event[node_name]
                
                # 2. ONLY yield if the node is 'finalize_response'
                # This ensures we only send the "final polished" version
                if node_name == "finalize_response":
                    text_output = node_data.get("final_response") or node_data.get("raw_response")
                    
                    if text_output:
                        payload = {
                            "text": text_output,
                            "session_id": active_session_id
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
            
            # 3. Final 'done' signal
            yield f"data: {json.dumps({'done': True, 'session_id': active_session_id})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': 'Streaming interrupted', 'session_id': active_session_id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")