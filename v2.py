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
import tab

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
TALIAN_KASIH = "Talian Kasih 15999 (call, 24 hours) or WhatsApp 019-261 5999"
BEFRIENDERS = "Befrienders 03-7627 2929 (24 hours)"
OSCC = "the emergency department of the nearest government hospital, and ask for the One Stop Crisis Centre (OSCC)"
TRIBUNAL = "Tribunal for Anti-bully (free and private): tab.bheuu.gov.my or 03-2271 1000"

COUNSELLOR_PROMPT = f"""You are Berani Counsellor in a Malaysian anti bullying app used mostly by school students aged 10 to 18, and sometimes their parents. You are warm, steady and practical. Your job is to make sure the person gets real help from real people, not just sympathy.

Every reply MUST follow this structure, in plain text (no markdown, no asterisks, no headings):

1. Acknowledge in one or two sentences. Name it plainly when it is bullying ("What they did is bullying, and it is not your fault.").
2. Safety. If anyone was hit, slapped, pushed, touched sexually, threatened, or something was put on or in their body or clothes: ask "Are you safe right now? Are you hurt?" If hurt, tell them to see a doctor today, and for serious injury or anything sexual to go to {OSCC}. If in danger now: call 999.
3. A numbered action plan of 3 or 4 specific steps, for example:
   - Tell a trusted adult today: a parent, the class teacher or the school counsellor. Suggest the exact words to use ("I need help. Someone at school ... ").
   - Write it down while it is fresh: date, time, place, who did it, who saw. Keep screenshots. Use Berani's "Record what happened".
   - For bullying at school: make a written complaint to the school's anti-bully committee (every school must have one under the Anti-bully Act 2026). If the school does nothing or handles it badly, the student or a parent can go to the {TRIBUNAL}. If it happened outside school, a child can go to the Tribunal directly.
   - For assault, sexual harassment, threats or demands for money: a police report can be made too.
4. End with: "If you need to talk to someone now, call {TALIAN_KASIH}." Then one short, caring question to keep the conversation going.

Other rules:
- Be specific and concrete. Never give a vague or one-line answer.
- Any mention of suicide, self harm, wanting to die or disappear: stop the plan. Say clearly that their life matters, urge them to call {BEFRIENDERS} or {TALIAN_KASIH} now, or 999 if they might act on it, and to stay with a trusted adult.
- Reply in the user's language (Bahasa Melayu or English). Keep it readable on a phone: about 120 to 200 words.
- You are an AI, not a doctor, lawyer or emergency service. Never diagnose, never promise outcomes, never claim Berani will contact anyone.
- Do not ask for full names, addresses or school names of other children.
- Stay on safety, bullying, wellbeing and next steps. Politely decline unrelated tasks.

The Tribunal is the authority on its own process, not you:
- For facts about the Tribunal, use ONLY the TRIBUNAL FACTS below, and say they come from the Tribunal's website.
- Never decide or predict whether a case qualifies, what the Tribunal will decide, deadlines, compensation, or a complaint's status, and never give legal advice. For any of these, say this is a question for the Tribunal itself, and suggest asking through Tanya Kami ({tab.LINKS['askUs']}), which replies within 3 working days, or calling {tab.PHONE}. Offer to help them word the question, without names or IC numbers.
- Berani helps them prepare: record what happened and get Borang 1 ready. They report on the official portal, tab.bheuu.gov.my.

TRIBUNAL FACTS (from tab.bheuu.gov.my):
{tab.facts_block()}"""


def _has(t: str, words: list[str]) -> bool:
    import re
    return any(re.search(r"(^|[^a-z])" + re.escape(w.rstrip("~")) + ("" if w.endswith("~") else r"(?![a-z])"), t) for w in words)


SELF_HARM = ["kill myself", "end my life", "want to die", "wanna die", "suicid~", "self harm", "self-harm", "hurt myself",
             "cut myself", "disappear forever", "bunuh diri", "nak mati", "mahu mati", "ingin mati", "tak nak hidup", "cederakan diri"]
DANGER = ["kill you", "kill me", "knife", "gun", "weapon", "stab~", "follow me home", "followed me home",
          "bunuh", "pisau", "parang", "senjata", "tikam", "ikut saya balik"]
SEXUAL = ["rape~", "molest~", "touched me", "touch me", "private part~", "naked", "nude~", "sexual~", "pants", "underwear",
          "rogol", "cabul", "raba", "bogel", "seluar dalam", "gangguan seksual"]
PHYSICAL = ["slap~", "hit me", "hits me", "punch~", "kick~", "push~", "pushed", "beat~", "hurt me", "injur~", "bleed~", "bruise~",
            "choke~", "ants", "threw", "spit~", "tampar", "pukul", "tumbuk", "tendang", "tolak", "cedera", "lebam", "berdarah", "ludah", "semut"]
OFFICIAL = ["eligible", "qualify", "qualif~", "layak", "deadline", "tarikh akhir", "how long", "berapa lama", "status",
            "compensation", "pampasan", "ganti rugi", "lawyer", "peguam", "sue", "saman", "court", "mahkamah", "hearing",
            "perbicaraan", "decision", "keputusan", "appeal", "rayuan", "legal", "undang-undang", "outcome", "e-tab", "etab",
            "borang", "lampiran", "can i file", "can i report to the tribunal", "boleh failkan", "boleh adu"]
SCHOOL = ["school", "class", "classmate~", "teacher", "student", "hostel", "asrama", "recess", "canteen", "senior~", "form ",
          "sekolah", "kelas", "cikgu", "guru", "murid", "pelajar", "rehat", "kantin", "bully~", "buli"]


def screen(text: str) -> dict:
    t = f" {(text or '').lower()} "
    return {
        "selfHarm": _has(t, SELF_HARM),
        "danger": _has(t, DANGER),
        "sexual": _has(t, SEXUAL),
        "physical": _has(t, PHYSICAL),
        "school": _has(t, SCHOOL),
        "official": _has(t, OFFICIAL),
    }


def enforce(reply: str, flags: dict) -> str:
    """Deterministic floor: whatever the model wrote, these lines are always present."""
    out = reply.strip()
    low = out.lower()
    head = []
    if (flags["danger"] or flags["selfHarm"]) and "999" not in low:
        head.append("If you are in danger right now, call 999 or go to a trusted adult immediately.")
    if flags["selfHarm"] and "befrienders" not in low:
        head.append(f"Your life matters. Please call {BEFRIENDERS} or {TALIAN_KASIH} now.")
    tail = []
    if (flags["physical"] or flags["sexual"] or flags["danger"]) and "oscc" not in low and "one stop crisis" not in low:
        tail.append(f"If you are hurt, see a doctor today. For serious injury or anything sexual, go to {OSCC}.")
    if flags.get("official") and not flags["selfHarm"]:
        if "tanya-kami" not in low and "tanya kami" not in low:
            tail.append(f"For an official answer, ask the Tribunal itself: Tanya Kami at {tab.LINKS['askUs']} (they reply within 3 working days) or call {tab.PHONE}.")
    elif not flags["selfHarm"] and "tribunal" not in low:
        tail.append(f"If the school does not act, you or your parent can go to the {TRIBUNAL}.")
    if "15999" not in low:
        tail.append(f"If you need to talk to someone now, call {TALIAN_KASIH}.")
    return "\n\n".join([*head, out, *tail]).strip()


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

    # Flags come from everything the user has said, so earlier risks are not forgotten.
    flags = screen(" ".join(m["content"] for m in clean if m["role"] == "user"))

    if ai.provider() is None:
        return jsonify({"error": "ai_unavailable", "flags": flags}), 503
    try:
        reply = ai.chat(COUNSELLOR_PROMPT, clean, max_tokens=700, temperature=0.4)
        return jsonify({"reply": enforce(reply, flags), "flags": flags})
    except Exception as e:
        print("v2 chat error:", type(e).__name__)
        return jsonify({"error": "chat_failed", "flags": flags}), 502


@bp.get("/health")
def health():
    return jsonify({"ok": True, "version": 2, "ai": ai.provider()})
