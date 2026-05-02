"""OpenAI-backed tools for graph nodes."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from graph.prompts.persona import FALLBACK_MESSAGE, FINALIZER_PROMPT, QUALITY_PROMPT

OPENAI_API_KEY_ENV = "PUBLIC_AGENT_OPENAI_APIKEY"
DEFAULT_MODEL = "gpt-4o-mini"


def get_openai_client() -> OpenAI:
    """Create the OpenAI client used by the graph."""
    load_dotenv()
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{OPENAI_API_KEY_ENV} not set")
    return OpenAI(api_key=api_key)


def finalize_response(
    query: str,
    raw_response: str,
    route: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Supervisor final pass over raw node output."""
    if route in {"blocked", "clarify", "fallback_human"}:
        return raw_response

    if raw_response.strip() == FALLBACK_MESSAGE:
        return FALLBACK_MESSAGE

    prompt = f"""{FINALIZER_PROMPT}

User query:
{query}

Raw response:
{raw_response}
"""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0.6,
        max_tokens=700,
        messages=[
            {"role": "system", "content": FINALIZER_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or raw_response


def check_response_quality(
    query: str,
    final_response: str,
    route: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Judge whether a response should be accepted, retried, or escalated."""
    if route in {"blocked", "clarify", "cached_answer"}:
        return {"status": "accepted", "reason": "terminal route", "confidence": 1.0}
    if not final_response or not final_response.strip():
        return {"status": "retry", "reason": "empty response", "confidence": 0.0}

    if final_response.strip() == FALLBACK_MESSAGE:
        return {"status": "accepted", "reason": "approved fallback", "confidence": 1.0}

    prompt = f"""{QUALITY_PROMPT}
User query:
{query}

Route:
{route}

Response:
{final_response}
"""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0.6,
        max_tokens=160,
        messages=[
            {"role": "system", "content": "You are a strict quality gate. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    parsed = _parse_json_object(response.choices[0].message.content or "{}")
    status = parsed.get("status", "accepted")
    if status not in {"accepted", "retry", "fallback_human"}:
        status = "accepted"

    return {
        "status": status,
        "reason": str(parsed.get("reason", "")),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
    }


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}
