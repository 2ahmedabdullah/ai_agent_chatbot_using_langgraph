"""Hot-memory tools used by graph nodes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from caching.hot_memory import add_chat_message, load_user_chat_history


def load_hot_memory(session_id: str, user_id: Optional[str] = None, limit: int = 2) -> List[Dict[str, Any]]:
    """Load recent session chat history."""
    return load_user_chat_history(session_id=session_id, user_id=user_id, limit=limit)


def store_turn(session_id: str, query: str, response: str, user_id: Optional[str] = None) -> None:
    """Store the user and assistant turn in hot memory."""
    add_chat_message(session_id, "user", query, user_id=user_id)
    add_chat_message(session_id, "assistant", response, user_id=user_id)
