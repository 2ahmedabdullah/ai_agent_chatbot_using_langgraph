"""RAG subgraph for document-grounded answering."""

from __future__ import annotations

from typing import Any, Dict, List
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from rag.generator import GeneratorState, ResponseGenerator
from rag.retriever import RAGDocumentRetriever


class RAGState(TypedDict):
    """State used inside the RAG subgraph."""

    query: str
    chat_history: NotRequired[List[Dict[str, Any]]]
    retrieved_context: NotRequired[List[Dict[str, Any]]]
    raw_response: NotRequired[str]
    metadata: NotRequired[Dict[str, Any]]
    top_k: NotRequired[int]


def retrieve_context_node(state: RAGState) -> dict:
    retriever = RAGDocumentRetriever()
    context = retriever.retrieve(
        query=state["query"],
        top_k=state.get("top_k", 5),
    )
    return {
        "retrieved_context": context,
        "metadata": {
            **dict(state.get("metadata") or {}),
            "retrieved_context_count": len(context),
        },
    }


def generate_answer_node(state: RAGState) -> dict:
    generator = ResponseGenerator()
    context_texts = [item.get("content", "") for item in state.get("retrieved_context", [])]
    result = generator.generate(
        GeneratorState(
            query=state["query"],
            retrieved_context=context_texts,
            chat_history=state.get("chat_history", []),
        )
    )
    return {
        "raw_response": result.generated_response,
        "metadata": {
            **dict(state.get("metadata") or {}),
            **result.metadata,
        },
    }


def build_rag_graph():
    workflow = StateGraph(RAGState)
    retry_policy = RetryPolicy(max_attempts=2, initial_interval=1.0)

# We are now telling these nodes to use the 'retry_policy' defined above
    workflow.add_node("retrieve_context", retrieve_context_node, retry=retry_policy)    
    workflow.add_node("generate_answer", generate_answer_node, retry=retry_policy)

    workflow.add_edge(START, "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", END)
    return workflow.compile()


rag_graph = build_rag_graph()


def invoke_rag(query: str, chat_history: List[Dict[str, Any]] | None = None, top_k: int = 5) -> Dict[str, Any]:
    """Invoke the RAG subgraph from the main public-agent graph."""
    return rag_graph.invoke(
        {
            "query": query,
            "chat_history": chat_history or [],
            "top_k": top_k,
            "metadata": {},
        }
    )
