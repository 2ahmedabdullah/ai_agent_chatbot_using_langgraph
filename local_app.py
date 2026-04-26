"""
Local runner for the Public Agent Chatbot.

Usage:
- CLI:    python local_app.py
- Server: python local_app.py --server

FastAPI endpoints:
- GET  /health
- POST /chat
- POST /chat/stream
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging. WARNING)

load_dotenv()

from graph.builder import graph, stream_graph_events

app = FastAPI(
    title="Public Agent Chatbot",
    version="0.1.0",
    description="public-agent chatbot API backed by LangGraph, MongoDB cache, Qdrant RAG, and OpenAI.",
)


class ChatRequest(BaseModel):
    """Request body for public-agent chat."""

    query: str = Field(..., min_length=1, description="User query")
    session_id: Optional[str] = Field(default=None, description="Conversation/session id")
    user_id: Optional[str] = Field(default=None, description="Optional user id")
    hot_memory_limit: int = Field(default=2, ge=0, le=10)
    rag_top_k: int = Field(default=5, ge=1, le=10)
    cache_final_answer: bool = True


class ChatResponse(BaseModel):
    """Response returned by the public-agent chat endpoint."""

    answer: str
    session_id: Optional[str]
    route: Optional[str]
    cache_hit: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run the full public-agent graph and return the final answer."""
    result = graph.invoke(_request_to_state(request))
    return _state_to_response(result)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream graph node updates as server-sent events."""

    def event_stream():
        final_state: Dict[str, Any] = {}
        for event in stream_graph_events(_request_to_state(request)):
            final_state.update(_flatten_event(event))
            yield f"data: {json.dumps(event, default=str)}\n\n"

        if final_state:
            response = _state_to_response(final_state).model_dump()
            yield f"data: {json.dumps({'final': response}, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
        answer=state.get("final_response") or state.get("raw_response") or "",
        session_id=state.get("session_id"),
        route=state.get("route"),
        cache_hit=bool(state.get("cache_hit", False)),
        metadata=state.get("metadata") or {},
    )


def _flatten_event(event: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for value in event.values():
        if isinstance(value, dict):
            flattened.update(value)
    return flattened


def run_cli() -> None:
    """Interactive CLI test harness for the full chatbot."""
    print("--- Public Agent Chatbot CLI ---")
    print("Type 'quit', 'exit', or 'bye' to stop.")
    session_id: Optional[str] = None

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"quit", "exit", "bye"}:
            print("Bot: Goodbye. Take care.")
            return
        if not query:
            continue

        result = graph.invoke(
            {
                "user_query": query,
                "session_id": session_id,
                "hot_memory_limit": 2,
                "rag_top_k": 5,
                "cache_final_answer": True,
                "retry_count": 0,
                "max_retries": 1,
                "metadata": {},
            }
        )
        session_id = result.get("session_id") or session_id
        print(f"Bot: {result.get('final_response') or result.get('raw_response') or ''}")
        # print(f"[route={result.get('route')} cache_hit={result.get('cache_hit', False)} session_id={session_id}]")


def run_server() -> None:
    import uvicorn

    uvicorn.run("local_app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the public-agent chatbot locally.")
    parser.add_argument("--server", action="store_true", help="Run FastAPI server instead of CLI.")
    args = parser.parse_args()

    if args.server:
        run_server()
    else:
        run_cli()
