"""
LangChain answer generation for the RAG portion of the public agent.

The parent public-agent LangGraph should call ResponseGenerator from its
generate_response node after classify_query decides whether RAG is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from graph.prompts.persona import FALLBACK_MESSAGE, RAG_SYSTEM_PROMPT
except ImportError:
    FALLBACK_MESSAGE = (
        "I sincerely apologize, but I don't have the information needed to answer your request at this time. "
        "We are currently working hard to expand my capabilities and knowledge base to serve you better. "
        "For now, please reach out to **support@konguess.com**, and our team will be happy to assist you "
        "manually with this feature or data point."
    )
    RAG_SYSTEM_PROMPT = (
        "You are the Mesa public assistant. Answer only from retrieved context. "
        "If context is insufficient, use the fallback message."
    )

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 900
    top_p: float = 0.95


@dataclass
class GeneratorState:
    query: str
    retrieved_context: List[str]
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    generated_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseGenerator:
    """Use ChatOpenAI to synthesize an answer from retrieved context."""

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self.api_key = os.getenv("PUBLIC_AGENT_OPENAI_APIKEY")

        if not self.api_key:
            raise ValueError("PUBLIC_AGENT_OPENAI_APIKEY not set")

        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            api_key=self.api_key,
            streaming=False,
        )
        self.streaming_llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            api_key=self.api_key,
            streaming=True,
        )

    def generate(self, state: GeneratorState, system_prompt: Optional[str] = None) -> GeneratorState:
        if not state.retrieved_context:
            state.generated_response = FALLBACK_MESSAGE
            state.metadata = {
                "model": self.config.model_name,
                "context_count": 0,
                "generated_at": datetime.now().isoformat(),
                "fallback": True,
            }
            return state

        messages = self._build_messages(
            query=state.query,
            context=state.retrieved_context,
            history=state.chat_history,
            system_prompt=system_prompt,
        )

        response = self.llm.invoke(messages)
        answer = str(response.content or "")
        state.generated_response = answer
        state.metadata = {
            "model": self.config.model_name,
            "context_count": len(state.retrieved_context),
            "generated_at": datetime.now().isoformat(),
        }

        return state

    async def generate_streaming(
        self,
        state: GeneratorState,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        messages = self._build_messages(
            query=state.query,
            context=state.retrieved_context,
            history=state.chat_history,
            system_prompt=system_prompt,
        )

        async for chunk in self.streaming_llm.astream(messages):
            token = str(chunk.content or "")
            if token:
                yield token

    def _build_messages(
        self,
        query: str,
        context: List[str],
        history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt or self._default_system_prompt())
        ]

        for item in history[-6:]:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))

        context_text = "\n\n---\n\n".join(context).strip()
        messages.append(
            HumanMessage(
                content=(
                    "Use the retrieved context to answer the question. "
                    "If the answer is not in the context, return the fallback message exactly.\n\n"
                    f"[Fallback Message]\n{FALLBACK_MESSAGE}\n\n"
                    f"[Retrieved Context]\n{context_text or 'No relevant context was retrieved.'}\n\n"
                    f"[Question]\n{query}"
                )
            )
        )
        return messages

    @staticmethod
    def _default_system_prompt() -> str:
        return RAG_SYSTEM_PROMPT
