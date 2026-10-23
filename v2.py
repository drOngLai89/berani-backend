"""
Berani API v2.

Additive endpoints for Berani 3. The v1 routes in server.py stay untouched so the
app version currently in the stores keeps working.

Design rules:
- Stateless. Nothing a user sends is stored or logged (only sizes and error types).
- The system prompt lives here, never on the client, so the endpoint cannot be
  turned into a general purpose chatbot on our AI key.
- Hard caps on payload size, message count and request rate.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from flask import Blueprint, jsonify, request

import ai

bp = Blueprint("v2", __name__, url_prefix="/v2")

MAX_TEXT = 6000            # chars of incident text
MAX_IMAGES = 3
MAX_IMAGE_B64 = 1_500_000  # ~1.1 MB per image after base64
MAX_CHAT_MESSAGES = 30
MAX_CHAT_CHARS = 2000      # per message

# ---------------------------------------------------------------- rate limiting
_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_lock = threading.Lock()
LIMITS = {"analyse": (20, 3600), "chat": (80, 3600)}


def client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "unknown"


def rate_limited(bucket: str) -> bool:
    limit, window = LIMITS[bucket]
    now = time.time()
    key = (bucket, client_ip())
    with _lock:
        q = _hits[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            return True
        q.append(now)
        return False


def reset_rate_limits():  # used by tests
    with _lock:
        _hits.clear()


# ---------------------------------------------------------------- analysis
FORMS = ["physical", "verbal", "social", "discrimination", "online"]
HARMS = ["physical", "psychological", "social"]
SEVERITY = ["low", "moderate", "high", "critical"]

ANALYSIS_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "severity": {"type": "string", "enum": SEVERITY},
            "forms": {"type": "array", "items": {"type": "string", "enum": FORMS}},
            "harms": {"type": "array", "items": {"type": "string", "enum": HARMS}},
            "possibleCrime": {"type": "boolean"},
            "crimeNotes": {"type": "string"},
            "selfHarmRisk": {"type": "boolean"},
            "summary": {"type": "string"},
            "evidenceNotes": {"type": "string"},
            "missing": {"type": "array", "items": {"type": "string"}},
            "nextStep": {"type": "string"},
        },
        "required": [
            "severity", "forms", "harms", "possibleCrime", "crimeNotes",
            "selfHarmRisk", "summary", "evidenceNotes", "missing", "nextStep",
        ],
}

ANALYSIS_PROMPT = """You help young people in Malaysia record bullying clearly. You are not a lawyer and you never decide guilt.

Read the incident description and any screenshots or photos. Return JSON only.

Classify using the Anti-bully Act 2026 (Act 876), s. 3(2):
- physical: harm to the victim or their belongings
- verbal: insulting, threatening or demeaning words
- social: exclusion, damaging reputation, hostile environment
- discrimination: shaming or discrimination by gender, race, religion or disability
- online: any of the above through phones, apps, social media or messages
Harms: physical, psychological, social.

severity:
- low: unkind but limited, no threat
- moderate: hurtful, humiliating, or repeated
- high: threats, injury, sharing private images, extortion, sustained campaign
- critical: the person may be in danger now or soon (weapons, threats to kill or seriously hurt, sexual assault, being followed, or any sign of suicide or self harm)

possibleCrime: true if it may involve assault, sexual harassment or assault, criminal intimidation, extortion, sharing intimate images, or stalking. crimeNotes: one short plain sentence naming what it may involve, or "" if none.
selfHarmRisk: true only if the writer suggests they may hurt themselves or end their life.
summary: 2 to 4 neutral, factual sentences in the third person ("The student reports..."). Facts only, no opinions, no names beyond what was written. Same language as the description (Malay or English).
evidenceNotes: what the screenshots/photos show, factually, or "" if none were given. Never describe sexual content in detail.
missing: up to 4 short questions for details that would make this record stronger (when, where, who saw it, how often).
nextStep: one warm, practical sentence for a teenager about what to do next. If severity is critical, tell them to get to a safe adult or call 999 now.

Write simply, for a 13 to 17 year old. Never ask for or repeat full names of other children unless already given."""


def _validate_analysis(d: dict) -> dict:
    out = {
        "severity": d.get("severity") if d.get("severity") in SEVERITY else "moderate",
        "forms": [f for f in d.get("forms", []) if f in FORMS],
        "harms": [h for h in d.get("harms", []) if h in HARMS],
        "possibleCrime": bool(d.get("possibleCrime")),
        "crimeNotes": str(d.get("crimeNotes", ""))[:300],
        "selfHarmRisk": bool(d.get("selfHarmRisk")),
        "summary": str(d.get("summary", ""))[:1200],
        "evidenceNotes": str(d.get("evidenceNotes", ""))[:800],
        "missing": [str(m)[:160] for m in d.get("missing", [])][:4],
        "nextStep": str(d.get("nextStep", ""))[:300],
    }
    # Safety floor: self harm risk is always treated as critical.
    if out["selfHarmRisk"]:
        out["severity"] = "critical"
    return out


@bp.post("/analyse")
def analyse():
    if rate_limited("analyse"):
        return jsonify({"error": "rate_limited"}), 429
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or "").strip()
    images = data.get("images") or []
    if not isinstance(images, list):
        return jsonify({"error": "bad_images"}), 400
    if not text and not images:
        return jsonify({"error": "empty"}), 400
    if len(text) > MAX_TEXT:
        return jsonify({"error": "text_too_long"}), 413
    if len(images) > MAX_IMAGES or any(
        not isinstance(i, str) or len(i) > MAX_IMAGE_B64 for i in images
    ):
        return jsonify({"error": "images_too_large"}), 413

    if ai.provider() is None:
        return jsonify({"error": "ai_unavailable"}), 503

    prompt = f"Incident date: {str(data.get('date') or 'not given')[:20]}\n\nDescription:\n{text or '(no text, see images)'}"
    try:
        raw = ai.structured(ANALYSIS_PROMPT, prompt, images, "incident_analysis", ANALYSIS_SCHEMA)
        return jsonify({"analysis": _validate_analysis(raw)})
    except Exception as e:  # never log user content
        print("v2 analyse error:", type(e).__name__)
        return jsonify({"error": "analysis_failed"}), 502


# ---------------------------------------------------------------- counsellor
COUNSELLOR_PROMPT = """You are Berani Counsellor, a warm, calm support companion in a Malaysian anti bullying app, mostly used by teenagers.

How you help:
- Listen first. Reflect feelings briefly. Keep replies short: 3 to 6 short sentences or a few bullets.
- Give practical, age appropriate next steps: tell a trusted adult, keep evidence (screenshots with dates), stay with friends, block and report online.
- You can explain the Anti-bully Act 2026: for bullying at school, complain to the school's anti bully committee first; if they take no action or handle it badly, the family can go to the Tribunal for Anti-bully (free, private, at tab.bheuu.gov.my). Children bullied outside school can go to the Tribunal directly. Only the victim or a parent or guardian can file. Mention the Case file feature in the app, which builds the paperwork.
- Reply in the user's language (Malay or English).

Safety rules (always):
- If there is any sign of immediate danger, weapons, sexual assault, or suicide or self harm: put safety first. Say clearly to call 999 or go to a trusted adult now. For suicide or self harm thoughts also give Befrienders 03-7627 2929 (24/7) and Talian Kasih 15999.
- For sexual assault: the nearest hospital emergency department has a One Stop Crisis Centre (OSCC). Do not give graphic detail.
- You are not a doctor, lawyer or emergency service. Never diagnose. Never promise outcomes. Never claim Berani will contact anyone or investigate.
- Do not ask for full names, addresses or school names of other children.
- Stay on the topic of safety, bullying, wellbeing and next steps. Politely decline unrelated tasks (homework, coding, essays).
- Plain text only: no markdown headings, no asterisks."""


@bp.post("/chat")
def chat():
    if rate_limited("chat"):
        return jsonify({"error": "rate_limited"}), 429
    data = request.get_json(silent=True) or {}
    msgs = data.get("messages") or []
    if not isinstance(msgs, list) or not msgs:
        return jsonify({"error": "empty"}), 400

    clean = []
    for m in msgs[-MAX_CHAT_MESSAGES:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):  # client system prompts are ignored
            continue
        content = str(m.get("content") or "")[:MAX_CHAT_CHARS]
        if content.strip():
            clean.append({"role": role, "content": content})
    if not clean or clean[-1]["role"] != "user":
        return jsonify({"error": "empty"}), 400

    if ai.provider() is None:
        return jsonify({"error": "ai_unavailable"}), 503
    try:
        return jsonify({"reply": ai.chat(COUNSELLOR_PROMPT, clean, max_tokens=450, temperature=0.5)})
    except Exception as e:
        print("v2 chat error:", type(e).__name__)
        return jsonify({"error": "chat_failed"}), 502


@bp.get("/health")
def health():
    return jsonify({"ok": True, "version": 2, "ai": ai.provider()})
