from __future__ import annotations

from pathlib import Path
import types
from typing import Any, Dict, List

import pytest
import sys
import importlib.util
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_test_doubles_for_missing_deps() -> None:
    """Install lightweight module stubs so tests can run without cloud deps."""
    if (
        "langgraph.graph" not in sys.modules
        and importlib.util.find_spec("langgraph") is None
    ):
        langgraph_mod = types.ModuleType("langgraph")
        langgraph_graph = types.ModuleType("langgraph.graph")
        langgraph_types = types.ModuleType("langgraph.types")

        START = "__start__"
        END = "__end__"

        class RetryPolicy:
            def __init__(self, max_attempts=1, initial_interval=0.0):
                self.max_attempts = max_attempts
                self.initial_interval = initial_interval

        class _CompiledGraph:
            def __init__(self, nodes, entry_point, edges, conditional):
                self._nodes = nodes
                self._entry_point = entry_point
                self._edges = edges
                self._conditional = conditional

            def invoke(self, state):
                graph_state = dict(state)
                current = self._entry_point
                while current and current != END:
                    update = self._nodes[current](graph_state) or {}
                    if isinstance(update, dict):
                        graph_state.update(update)
                    if current in self._conditional:
                        selector, mapping = self._conditional[current]
                        current = mapping.get(selector(graph_state))
                    else:
                        current = (self._edges.get(current) or [None])[0]
                return graph_state

            def stream(self, state):
                graph_state = dict(state)
                current = self._entry_point
                while current and current != END:
                    update = self._nodes[current](graph_state) or {}
                    if isinstance(update, dict):
                        graph_state.update(update)
                    yield {current: update}
                    if current in self._conditional:
                        selector, mapping = self._conditional[current]
                        current = mapping.get(selector(graph_state))
                    else:
                        current = (self._edges.get(current) or [None])[0]

        class StateGraph:
            def __init__(self, _state_type):
                self.nodes = {}
                self.edges = {}
                self.conditional = {}
                self.entry_point = None

            def add_node(self, name, fn, retry=None):
                self.nodes[name] = fn

            def set_entry_point(self, name):
                self.entry_point = name

            def add_edge(self, source, target):
                self.edges.setdefault(source, []).append(target)

            def add_conditional_edges(self, source, selector, mapping):
                self.conditional[source] = (selector, mapping)

            def compile(self):
                entry = self.entry_point or (self.edges.get(START) or [None])[0]
                return _CompiledGraph(self.nodes, entry, self.edges, self.conditional)

        langgraph_graph.START = START
        langgraph_graph.END = END
        langgraph_graph.StateGraph = StateGraph
        langgraph_types.RetryPolicy = RetryPolicy
        sys.modules["langgraph"] = langgraph_mod
        sys.modules["langgraph.graph"] = langgraph_graph
        sys.modules["langgraph.types"] = langgraph_types

    if (
        "langchain_openai" not in sys.modules
        and importlib.util.find_spec("langchain_openai") is None
    ):
        langchain_openai = types.ModuleType("langchain_openai")

        class ChatOpenAI:
            def __init__(self, *args, **kwargs):
                pass

            def with_structured_output(self, _schema):
                return self

            def invoke(self, _messages):
                return types.SimpleNamespace(content="SAFE")

        class OpenAIEmbeddings:
            def __init__(self, *args, **kwargs):
                pass

        langchain_openai.ChatOpenAI = ChatOpenAI
        langchain_openai.OpenAIEmbeddings = OpenAIEmbeddings
        sys.modules["langchain_openai"] = langchain_openai

    if (
        "langchain_core.messages" not in sys.modules
        and importlib.util.find_spec("langchain_core") is None
    ):
        langchain_core = types.ModuleType("langchain_core")
        langchain_core_messages = types.ModuleType("langchain_core.messages")

        class _BaseMessage:
            def __init__(self, content=""):
                self.content = content

        class SystemMessage(_BaseMessage):
            pass

        class HumanMessage(_BaseMessage):
            pass

        class AIMessage(_BaseMessage):
            pass

        langchain_core_messages.BaseMessage = _BaseMessage
        langchain_core_messages.SystemMessage = SystemMessage
        langchain_core_messages.HumanMessage = HumanMessage
        langchain_core_messages.AIMessage = AIMessage
        sys.modules["langchain_core"] = langchain_core
        sys.modules["langchain_core.messages"] = langchain_core_messages

    if (
        "langchain_core.documents" not in sys.modules
        and importlib.util.find_spec("langchain_core") is None
    ):
        langchain_core_documents = types.ModuleType("langchain_core.documents")

        class Document:
            def __init__(self, page_content="", metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

        langchain_core_documents.Document = Document
        sys.modules["langchain_core.documents"] = langchain_core_documents

    if (
        "langchain_qdrant" not in sys.modules
        and importlib.util.find_spec("langchain_qdrant") is None
    ):
        langchain_qdrant = types.ModuleType("langchain_qdrant")

        class QdrantVectorStore:
            def __init__(self, *args, **kwargs):
                pass

            def add_documents(self, documents=None, ids=None):
                return None

            def similarity_search_with_score(self, query, k):
                return []

        langchain_qdrant.QdrantVectorStore = QdrantVectorStore
        sys.modules["langchain_qdrant"] = langchain_qdrant

    if (
        "qdrant_client" not in sys.modules
        and importlib.util.find_spec("qdrant_client") is None
    ):
        qdrant_client = types.ModuleType("qdrant_client")
        qdrant_http = types.ModuleType("qdrant_client.http")
        qdrant_models = types.ModuleType("qdrant_client.http.models")

        class _Collections:
            collections = []

        class QdrantClient:
            def __init__(self, *args, **kwargs):
                pass

            def get_collections(self):
                return _Collections()

            def create_collection(self, *args, **kwargs):
                return None

        class Distance:
            COSINE = "cosine"

        class VectorParams:
            def __init__(self, size, distance):
                self.size = size
                self.distance = distance

        qdrant_client.QdrantClient = QdrantClient
        qdrant_models.Distance = Distance
        qdrant_models.VectorParams = VectorParams
        sys.modules["qdrant_client"] = qdrant_client
        sys.modules["qdrant_client.http"] = qdrant_http
        sys.modules["qdrant_client.http.models"] = qdrant_models


# Opt-in fallback stubs for environments that do not install runtime deps.
# Default is disabled so live integration tests always use real packages.
if os.getenv("FORCE_TEST_DOUBLES", "0") == "1":
    _install_test_doubles_for_missing_deps()


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _load_rag_docs_snippets(limit: int = 3, chars_per_doc: int = 400) -> List[str]:
    docs_dir = Path(__file__).resolve().parents[1] / "rag" / "rag_docs"
    if not docs_dir.exists():
        return []

    snippets: List[str] = []
    for file_path in sorted(docs_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".txt", ".md"}:
            continue
        content = _read_text_file(file_path)
        if content:
            snippets.append(content[:chars_per_doc])
        if len(snippets) >= limit:
            break
    return snippets


@pytest.fixture
def base_state() -> Dict[str, Any]:
    """State fixture aligned with the API-to-graph shape in app.py."""
    return {
        "session_id": "test-session-001",
        "user_query": "What are Mesa public service hours?",
        "metadata": {},
        "retry_count": 0,
        "max_retries": 1,
        "rag_top_k": 5,
        "cache_final_answer": True,
    }


@pytest.fixture
def rag_docs_snippets() -> List[str]:
    """
    Small text excerpts from rag/rag_docs for grounding assertions.
    If the docs folder is not present in local dev, RAG grounding tests can skip.
    """
    return _load_rag_docs_snippets()
