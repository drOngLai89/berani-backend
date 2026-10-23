"""
AI provider layer. Anthropic (Claude) is used when ANTHROPIC_API_KEY is set,
otherwise OpenAI when OPENAI_API_KEY is set. Callers never touch a vendor SDK.
"""
from __future__ import annotations

import json
import os

ANTHROPIC_MODEL = os.getenv("BERANI_ANTHROPIC_MODEL", "claude-haiku-5-5")
OPENAI_MODEL = os.getenv("BERANI_MODEL", "gpt-4o-mini")
TIMEOUT = 45

_anthropic = None
_openai = None


class AIUnavailable(Exception):
    pass


def provider() -> str | None:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


def _anthropic_client():
    global _anthropic
    if _anthropic is None:
        import anthropic
        _anthropic = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=TIMEOUT)
    return _anthropic


def _openai_client():
    global _openai
    if _openai is None:
        from openai import OpenAI
        _openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=TIMEOUT)
    return _openai


def normalise(messages: list[dict]) -> list[dict]:
    """Starts with a user turn and alternates roles (merging repeats), as the Messages API expects."""
    out: list[dict] = []
    for m in messages:
        if not out and m["role"] != "user":
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def chat(system: str, messages: list[dict], max_tokens: int = 450, temperature: float = 0.5) -> str:
    """messages: [{'role': 'user'|'assistant', 'content': str}], last one from the user."""
    messages = normalise(messages)
    p = provider()
    if p == "anthropic":
        r = _anthropic_client().messages.create(
            model=ANTHROPIC_MODEL, max_tokens=max_tokens, temperature=temperature, system=system, messages=messages,
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
    if p == "openai":
        r = _openai_client().chat.completions.create(
            model=OPENAI_MODEL, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return (r.choices[0].message.content or "").strip()
    raise AIUnavailable()


def structured(system: str, text: str, images_b64: list[str], name: str, schema: dict) -> dict:
    """One user turn (text + JPEG images) -> JSON object matching `schema`."""
    p = provider()
    if p == "anthropic":
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}} for b in images_b64]
        content.append({"type": "text", "text": text})
        r = _anthropic_client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1200,
            temperature=0.2,
            system=system,
            tools=[{"name": name, "description": "Record the structured analysis.", "input_schema": schema}],
            tool_choice={"type": "tool", "name": name},
            messages=[{"role": "user", "content": content}],
        )
        for b in r.content:
            if getattr(b, "type", "") == "tool_use":
                return dict(b.input)
        raise ValueError("no tool_use block")
    if p == "openai":
        content = [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}", "detail": "low"}} for b in images_b64
        ]
        r = _openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        )
        return json.loads(r.choices[0].message.content or "{}")
    raise AIUnavailable()
