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


TRANSCRIBE_MODEL = os.getenv("BERANI_TRANSCRIBE_MODEL", "whisper-1")
# Segments Whisper itself rates as probably not speech, or very low confidence, are dropped:
# silence and noise must never turn into words in someone's complaint.
NO_SPEECH_MAX = 0.5
LOGPROB_MIN = -1.0


def can_transcribe() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def keep_segments(segments: list) -> str:
    out = []
    for seg in segments or []:
        get = (lambda k: seg.get(k)) if isinstance(seg, dict) else (lambda k: getattr(seg, k, None))
        nsp, lp, text = get("no_speech_prob") or 0.0, get("avg_logprob") or 0.0, (get("text") or "").strip()
        if text and nsp <= NO_SPEECH_MAX and lp >= LOGPROB_MIN:
            out.append(text)
    return " ".join(out).strip()


def _seg_get(seg, k):
    return seg.get(k) if isinstance(seg, dict) else getattr(seg, k, None)


def _confidence(segments: list) -> float:
    """Length-weighted mean avg_logprob of the segments kept: higher means Whisper is surer of the words."""
    total, weight = 0.0, 0
    for seg in segments or []:
        text = (_seg_get(seg, "text") or "").strip()
        if not text or (_seg_get(seg, "no_speech_prob") or 0.0) > NO_SPEECH_MAX:
            continue
        n = len(text)
        total += (_seg_get(seg, "avg_logprob") or -5.0) * n
        weight += n
    return total / weight if weight else -99.0


def _whisper(audio: bytes, filename: str, mime: str, language: str | None = None):
    kw = {"language": language} if language else {}
    r = _openai_client().audio.transcriptions.create(
        model=TRANSCRIBE_MODEL, file=(filename, audio, mime), response_format="verbose_json", temperature=0, **kw,
    )
    segments = getattr(r, "segments", None)
    lang = getattr(r, "language", None)
    if isinstance(r, dict):
        segments = r.get("segments", segments)
        lang = r.get("language", lang)
    return (lang or "").lower(), segments or []


# Whisper guesses the language first. With Malaysian speakers it sometimes guesses Malay for
# English (or English for Malay) and then writes a translation instead of what was said.
# For these languages, also transcribe with each one forced and keep the most confident result:
# a faithful transcript scores far higher than a translation.
LOCAL_LANGS = {"english": "en", "en": "en", "malay": "ms", "ms": "ms", "indonesian": "id", "id": "id"}


def transcribe(audio: bytes, filename: str, mime: str, language: str | None = None) -> str:
    """Speech to text with OpenAI Whisper, in the language spoken. No prompt (a prompt can be "heard"
    in silence). Audio is not stored. language: 'en' or 'ms' to force one, else detected."""
    if not can_transcribe():
        raise AIUnavailable()
    if language in ("en", "ms"):
        return keep_segments(_whisper(audio, filename, mime, language)[1])
    detected, segments = _whisper(audio, filename, mime)
    code = LOCAL_LANGS.get(detected)
    if not code:
        return keep_segments(segments)
    best, best_score = segments, _confidence(segments)
    for other in ("en", "ms"):
        if other == code:
            continue
        try:
            _, alt = _whisper(audio, filename, mime, other)
        except Exception:  # noqa: BLE001 - the first transcript still stands
            continue
        score = _confidence(alt)
        if score > best_score + 0.05:
            best, best_score = alt, score
    return keep_segments(best)


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


def structured(system: str, text: str, images_b64: list[str], name: str, schema: dict, max_tokens: int = 1200) -> dict:
    """One user turn (text + JPEG images) -> JSON object matching `schema`."""
    p = provider()
    if p == "anthropic":
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}} for b in images_b64]
        content.append({"type": "text", "text": text})
        r = _anthropic_client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
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
