"""Cache tools used by graph nodes."""

from __future__ import annotations

from typing import Any, Dict

from caching.cold_memory import check_cache, set_cache_result


def lookup_faq_cache(query: str) -> Dict[str, Any]:
    """Check exact/semantic FAQ cache."""
    return check_cache(query)


def store_faq_cache(query: str, response: str) -> None:
    """Store a successful final answer in the FAQ cache."""
    set_cache_result(query, response)
