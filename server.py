import os, textwrap, logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------------------------------------------------------
# OpenAI client (optional). If not available or key is bad, we fall back safely.
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
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

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Native apps don't need CORS, but harmless here.
logging.basicConfig(level=logging.INFO)

@app.get("/")
def root():
    return jsonify({"ok": True, "service": "berani-backend", "version": "1.2.0"})

@app.get("/health")
def health():
    return jsonify({"status": "healthy"})

@app.get("/diag")
def diag():
    """Quick diagnostics: shows whether OpenAI is configured."""
    return jsonify({
        "ok": True,
        "has_env_key": bool(OPENAI_API_KEY),
        "client_ready": bool(client),
        "openai_lib": openai_lib,
        "openai_error": openai_error,
    })

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

    # No OpenAI → return fallback immediately
    if not client:
        return jsonify({"answer": _fallback_answer(q)})

    try:
        # hard timeout so Render never hangs
        resp = client.responses.with_options(timeout=10).create(
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
        msg = _fallback_answer(q) + "\n\n(Note: AI had an issue; fallback shown.)"
        return jsonify({"answer": msg})

@app.post("/generate_report")
def generate_report():
    payload = request.get_json(force=True) or {}
    app.logger.info("POST /generate_report keys=%s", list(payload.keys()))
    if not client:
        return jsonify({"report": _fallback_report(payload)})

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
        resp = client.responses.with_options(timeout=12).create(
            model="gpt-4o-mini",
            input=prompt
        )
        text = resp.output_text or _fallback_report(payload)
        return jsonify({"report": text})
    except Exception as e:
        app.logger.exception("generate_report error: %s", e)
        return jsonify({"report": _fallback_report(payload)})

# Render start: gunicorn server:app -b 0.0.0.0:$PORT
