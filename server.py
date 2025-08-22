import os, textwrap, logging
from flask import Flask, request, jsonify
from flask_cors import CORS

DEBUG_FLAG = os.environ.get("DEBUG", "").lower() in {"1","true","yes","on"}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# OpenAI client (optional)
client = None
openai_lib = None
openai_error = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        openai_lib = "openai>=1.30 (responses API)"
    except Exception as e:
        client = None
        openai_error = f"{type(e).__name__}: {e}"

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

VERSION = "1.3.1"

@app.get("/")
def root():
    return jsonify({"ok": True, "service": "berani-backend", "version": VERSION})

@app.get("/health")
def health():
    return jsonify({"status": "healthy"})

@app.get("/diag")
def diag():
    return jsonify({
        "ok": True,
        "has_env_key": bool(OPENAI_API_KEY),
        "client_ready": bool(client),
        "openai_lib": openai_lib,
        "openai_error": openai_error,
        "version": VERSION,
    })

@app.get("/diag_openai")
def diag_openai():
    if not client:
        return jsonify({"ok": False, "why": "no_client", "error": openai_error})
    try:
        # NOTE: with_options is on the client
        resp = client.with_options(timeout=8).responses.create(
            model="gpt-4o-mini",
            input="Reply with the single word: OK"
        )
        text = (resp.output_text or "").strip()
        return jsonify({"ok": True, "model": "gpt-4o-mini", "text": text[:100]})
    except Exception as e:
        return jsonify({"ok": False, "why": "openai_error", "error": f"{type(e).__name__}: {e}"})

def _fallback_answer(_: str) -> str:
    return (
        "I’m sorry this happened. Here are immediate steps you can take:\n"
        "• If you’re in danger, move to a safe place and contact local authorities.\n"
        "• Seek medical attention if you’re hurt.\n"
        "• Write down what happened (who/what/when/where) and save any evidence.\n"
        "• Talk to a trusted adult or counselor.\n"
        "• Use the New Report tab to record details; attach photos if appropriate.\n"
        "When you’re ready, I can help you draft a clear, respectful incident report."
    )

def _fallback_report(payload: dict) -> str:
    return textwrap.dedent(f"""
    Incident Report (Draft)
    -----------------------
    Category: {payload.get('category') or '-'}
    Date:     {payload.get('dateISO') or '-'}
    Time:     {payload.get('timeISO') or '-'}
    Location: {payload.get('locationText') or '-'}

    Description:
    {payload.get('description') or '-'}

    Notes:
    • Keep language factual and neutral.
    • If any detail is approximate, say so (e.g., “about 11:40 AM”).
    • Attach photos or other evidence where safe and appropriate.
    """).strip()

@app.post("/assistant")
def assistant():
    data = request.get_json(force=True) or {}
    q = (data.get("question") or "").strip()
    app.logger.info("POST /assistant q=%r", q[:80])
    if not q:
        return jsonify({"answer": "Please enter a question."})

    if not client:
        return jsonify({"answer": _fallback_answer(q), "meta": {"fallback": True, "reason": "no_client"}})

    try:
        resp = client.with_options(timeout=10).responses.create(
            model="gpt-4o-mini",
            input=(
                "You are a supportive, non-judgmental safety assistant. "
                "Be concise, avoid stereotypes, and give practical steps.\n\n"
                f"Question: {q}"
            ),
        )
        answer = resp.output_text or _fallback_answer(q)
        return jsonify({"answer": answer})
    except Exception as e:
        app.logger.exception("assistant error: %s", e)
        meta = {"fallback": True}
        if DEBUG_FLAG: meta["error"] = f"{type(e).__name__}: {e}"
        msg = _fallback_answer(q) + "\n\n(Note: AI had an issue; fallback shown.)"
        return jsonify({"answer": msg, "meta": meta})

@app.post("/generate_report")
def generate_report():
    payload = request.get_json(force=True) or {}
    app.logger.info("POST /generate_report keys=%s", list(payload.keys()))

    if not client:
        return jsonify({"report": _fallback_report(payload), "meta": {"fallback": True, "reason": "no_client"}})

    prompt = textwrap.dedent(f"""
    Draft a clear, neutral incident report. Include who/what/when/where, impact,
    and (if appropriate) next steps. Avoid stereotypes or assumptions.

    Category: {payload.get('category')}
    Date ISO: {payload.get('dateISO')}
    Time ISO: {payload.get('timeISO')}
    Location: {payload.get('locationText')}
    Description: {payload.get('description')}
    """).strip()

    try:
        resp = client.with_options(timeout=12).responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        text = resp.output_text or _fallback_report(payload)
        return jsonify({"report": text})
    except Exception as e:
        app.logger.exception("generate_report error: %s", e)
        meta = {"fallback": True}
        if DEBUG_FLAG: meta["error"] = f"{type(e).__name__}: {e}"
        return jsonify({"report": _fallback_report(payload), "meta": meta})

# Start on Render:
# gunicorn server:app -b 0.0.0.0:$PORT
