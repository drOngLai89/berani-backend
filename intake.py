"""
Berani 3.1 intake: speak or type what happened, and the AI fills in the record and Borang 1.

- /v2/transcribe : voice note -> text (audio is never stored or logged)
- /v2/intake     : free text -> incidents for the user to confirm, plus questions for what is missing
- /v2/borang     : confirmed incidents -> draft wording for Borang 1 (facts, nature of bullying)
- /v2/config     : Tribunal links and server switches (e.g. email filing) for the app

The AI only organises what the user said. It never invents facts, names or dates; anything
not said stays empty and becomes a question. IC numbers, addresses and signatures never come here.
"""
from __future__ import annotations

import base64
import binascii
import re
from datetime import date

from flask import jsonify, request

import ai
import tab
from v2 import FORMS, HARMS, LIMITS, SEVERITY, bp, rate_limited, screen

LIMITS.update({"transcribe": (30, 3600), "intake": (30, 3600), "borang": (20, 3600)})

MAX_AUDIO_B64 = 4_500_000  # about 3 minutes of compressed speech
AUDIO_TYPES = {
    "audio/m4a": "m4a", "audio/x-m4a": "m4a", "audio/mp4": "m4a", "audio/aac": "aac",
    "audio/webm": "webm", "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
}
MAX_INTAKE_TEXT = 8000
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@bp.get("/config")
def config():
    return jsonify(tab.config())


@bp.post("/transcribe")
def transcribe():
    if rate_limited("transcribe"):
        return jsonify({"error": "rate_limited"}), 429
    data = request.get_json(silent=True) or {}
    mime = str(data.get("mime") or "audio/m4a").lower()
    b64 = data.get("audio")
    if mime not in AUDIO_TYPES:
        return jsonify({"error": "bad_audio_type"}), 400
    if not isinstance(b64, str) or not b64:
        return jsonify({"error": "empty"}), 400
    if len(b64) > MAX_AUDIO_B64:
        return jsonify({"error": "audio_too_long"}), 413
    try:
        audio = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return jsonify({"error": "bad_audio"}), 400
    if not ai.can_transcribe():
        return jsonify({"error": "ai_unavailable"}), 503
    try:
        text = ai.transcribe(audio, f"voice.{AUDIO_TYPES[mime]}", mime)
    except Exception as e:  # never log content
        print("v2 transcribe error:", type(e).__name__)
        return jsonify({"error": "transcribe_failed"}), 502
    return jsonify({"text": text[:MAX_INTAKE_TEXT], "flags": screen(text)})


# ---------------------------------------------------------------- intake
INCIDENT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "date": {"type": "string"},
        "time": {"type": "string"},
        "place": {"type": "string"},
        "whatHappened": {"type": "string"},
        "who": {"type": "string"},
        "witnesses": {"type": "string"},
        "atSchool": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "forms": {"type": "array", "items": {"type": "string", "enum": FORMS}},
        "harms": {"type": "array", "items": {"type": "string", "enum": HARMS}},
        "evidence": {"type": "string"},
    },
    "required": ["date", "time", "place", "whatHappened", "who", "witnesses", "atSchool", "forms", "harms", "evidence"],
}
INTAKE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "incidents": {"type": "array", "items": INCIDENT},
        "unsure": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
        "severity": {"type": "string", "enum": SEVERITY},
        "selfHarmRisk": {"type": "boolean"},
        "possibleCrime": {"type": "boolean"},
    },
    "required": ["incidents", "unsure", "questions", "severity", "selfHarmRisk", "possibleCrime"],
}

INTAKE_PROMPT = """You help a Malaysian student or parent turn what they said about bullying into a clear record. They may speak Bahasa Melayu, English or a mix. You organise; you never invent.

Return JSON only.

incidents: one entry per separate event. Split whenever something happened on a different day, in a different place, or in a different way
(for example, being pushed in the canteen yesterday and threatening WhatsApp messages today are TWO incidents). Never merge events that happened on different days.
- date: YYYY-MM-DD only if the person said or clearly implied it. Work out relative days ("yesterday", "semalam", "last Monday", "Isnin lepas") from TODAY. Otherwise "".
- time: HH:MM if said (e.g. "after school" is not a time), else "".
- place: exactly as they said it, without adding words, else "".
- whatHappened: their account in their own language, first person if they used it, close to their own words. Tidy only filler words and repetition. Never add details, motives, names or feelings they did not say.
- who: who did it, exactly as described (a name only if they said one), else "".
- witnesses: who saw it, if said, else "".
- atSchool: "yes" if it happened in school, class, hostel, on a school trip or school transport, or under school supervision; "no" if clearly outside; otherwise "unsure".
- forms (Anti-bully Act 2026 s.3(2)): physical, verbal, social, discrimination, online. harms: physical, psychological, social. Only what the account supports.
- evidence: things they say they have kept or can show (screenshots, photos, videos, messages, a police report), else "". People are witnesses, not evidence.
unsure: short labels for anything you had to guess (e.g. "date of the first incident").
questions: up to 3 short, gentle questions, in their language, ONLY about details that are still empty in your incidents, in this order of importance: when it happened, where, whether it was at school, who saw it, whether they kept screenshots. Never ask about something they already told you. Empty if nothing important is missing.
severity: low, moderate, high, or critical (danger now or soon: weapons, threats to kill or seriously hurt, sexual assault, being followed, or any sign of suicide or self harm).
selfHarmRisk: true only if they suggest they may hurt themselves or end their life.
possibleCrime: true if it may involve assault, sexual harassment or assault, criminal intimidation, extortion, sharing intimate images, or stalking."""


def _clean_incident(d: dict, today: str) -> dict:
    dt = str(d.get("date") or "")
    if not DATE_RE.match(dt) or dt > today:
        dt = ""
    tm = str(d.get("time") or "")
    if not re.match(r"^\d{2}:\d{2}$", tm):
        tm = ""
    return {
        "date": dt,
        "time": tm,
        "place": str(d.get("place") or "")[:200],
        "whatHappened": str(d.get("whatHappened") or "")[:4000],
        "who": str(d.get("who") or "")[:200],
        "witnesses": str(d.get("witnesses") or "")[:200],
        "atSchool": d.get("atSchool") if d.get("atSchool") in ("yes", "no", "unsure") else "unsure",
        "forms": [f for f in d.get("forms", []) if f in FORMS],
        "harms": [h for h in d.get("harms", []) if h in HARMS],
        "evidence": str(d.get("evidence") or "")[:300],
    }


_Q_TOPICS = {
    "date": ("bila", "when", "tarikh", "date", "hari apa", "which day"),
    "place": ("di mana", "mana", "where", "tempat", "place"),
    "witnesses": ("nampak", "saw", "saksi", "witness", "siapa lagi", "who else"),
    "atSchool": ("sekolah", "school"),
}


def _useful_questions(questions: list[str], incidents: list[dict]) -> list[str]:
    """Drop questions about details every incident already has, so nobody is asked twice."""
    answered = {
        "date": all(i["date"] for i in incidents),
        "place": all(i["place"] for i in incidents),
        "witnesses": all(i["witnesses"] for i in incidents),
        "atSchool": all(i["atSchool"] != "unsure" for i in incidents),
    }
    keep = []
    for q in questions:
        low = q.lower()
        if any(answered[t] and any(w in low for w in words) for t, words in _Q_TOPICS.items()):
            continue
        keep.append(q)
    return keep


@bp.post("/intake")
def intake():
    if rate_limited("intake"):
        return jsonify({"error": "rate_limited"}), 429
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or "").strip()
    today = str(data.get("today") or date.today().isoformat())[:10]
    if not DATE_RE.match(today):
        today = date.today().isoformat()
    if not text:
        return jsonify({"error": "empty"}), 400
    if len(text) > MAX_INTAKE_TEXT:
        return jsonify({"error": "text_too_long"}), 413
    flags = screen(text)
    if ai.provider() is None:
        return jsonify({"error": "ai_unavailable", "flags": flags}), 503
    try:
        raw = ai.structured(INTAKE_PROMPT, f"TODAY: {today}\n\nWhat they said:\n{text}", [], "bullying_record", INTAKE_SCHEMA, max_tokens=3000)
    except Exception as e:
        print("v2 intake error:", type(e).__name__)
        return jsonify({"error": "intake_failed", "flags": flags}), 502
    incidents = [_clean_incident(i, today) for i in (raw.get("incidents") or []) if isinstance(i, dict)][:8]
    if not incidents:  # never lose what they said
        incidents = [_clean_incident({"whatHappened": text}, today)]
    out = {
        "incidents": incidents,
        "questions": _useful_questions([str(q)[:160] for q in raw.get("questions", [])], incidents)[:3],
        "unsure": [str(u)[:80] for u in raw.get("unsure", [])][:6],
        "severity": raw.get("severity") if raw.get("severity") in SEVERITY else "moderate",
        "selfHarmRisk": bool(raw.get("selfHarmRisk")) or flags["selfHarm"],
        "possibleCrime": bool(raw.get("possibleCrime")),
        "flags": flags,
    }
    # Safety floor: anything that looks like danger or self harm is critical, whatever the model said.
    if out["selfHarmRisk"] or flags["danger"]:
        out["severity"] = "critical"
    return jsonify(out)


# ---------------------------------------------------------------- Borang 1 wording
BORANG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"facts": {"type": "string"}, "nature": {"type": "string"}},
    "required": ["facts", "nature"],
}

BORANG_PROMPT = """You draft two parts of Borang 1 (Aduan Buli), the complaint form of Malaysia's Tribunal for Anti-bully, from records the complainant has already confirmed. The complainant will read and edit your draft before using it.

facts ("Fakta-fakta buli"): the incidents in date order, as numbered short paragraphs. Each: date (and time if known), place, what happened, who did it and who saw it, exactly as recorded. Plain, calm, factual sentences. Written by the complainant: first person ("Saya" / "I") if the victim is filing; if a parent or guardian is filing, refer to "anak saya" / "my child". Add nothing that is not in the records: no motives, opinions, legal conclusions, or guesses. Where a detail is missing, leave it out; do not write "unknown".
nature ("Sifat perbuatan buli"): one short phrase naming the kinds of bullying in the records (physical, verbal, social, discrimination, online/cyber) and the harm (physical, psychological, social).

Write both in the requested language. No markdown."""


@bp.post("/borang")
def borang():
    if rate_limited("borang"):
        return jsonify({"error": "rate_limited"}), 429
    data = request.get_json(silent=True) or {}
    incidents = data.get("incidents") or []
    if not isinstance(incidents, list) or not incidents:
        return jsonify({"error": "empty"}), 400
    lang = "Bahasa Melayu" if data.get("language") == "ms" else "English"
    filer = "a parent or guardian of the child victim" if data.get("filer") == "parent" else "the victim"
    lines = []
    for i, inc in enumerate(incidents[:20], 1):
        if not isinstance(inc, dict):
            continue
        parts = [f"{k}: {str(inc.get(k) or '')[:1500]}" for k in ("date", "time", "place", "whatHappened", "who", "witnesses") if inc.get(k)]
        kinds = ", ".join([*map(str, inc.get("forms") or []), *map(str, inc.get("harms") or [])])
        lines.append(f"Incident {i}\n" + "\n".join(parts) + (f"\nkinds: {kinds}" if kinds else ""))
    body = "\n\n".join(lines)
    if len(body) > 20000:
        return jsonify({"error": "text_too_long"}), 413
    if ai.provider() is None:
        return jsonify({"error": "ai_unavailable"}), 503
    try:
        raw = ai.structured(BORANG_PROMPT, f"Language: {lang}\nFiled by: {filer}\n\nRecords:\n{body}", [], "borang_draft", BORANG_SCHEMA, max_tokens=3000)
    except Exception as e:
        print("v2 borang error:", type(e).__name__)
        return jsonify({"error": "borang_failed"}), 502
    return jsonify({"facts": str(raw.get("facts") or "")[:8000], "nature": str(raw.get("nature") or "")[:300]})
