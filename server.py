import os, textwrap
from flask import Flask, request, jsonify
from flask_cors import CORS

# OpenAI client (optional)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None  # if lib missing or bad key, we'll fall back

app = Flask(__name__)
CORS(app)  # safe for mobile; adjust if you later add a web app

@app.get("/")
def root():
    return jsonify({"ok": True, "service": "berani-backend", "version": "1.1.0"})

@app.get("/health")
def health():
    return jsonify({"status": "healthy"})

def _ai_fallback_answer(q: str) -> str:
    # Short, supportive fallback if OpenAI unavailable
    return (
        "I’m sorry this happened. Here are immediate steps you can take:\n"
        "• If you’re in danger, move to a safe place and contact local authorities.\n"
        "• Seek medical attention if you’re hurt.\n"
        "• Write down what happened (who/what/when/where) and save any evidence.\n"
        "• Talk to a trusted adult or counselor.\n"
        "• Use the New Report tab to record details; attach photos if appropriate.\n"
        "When you’re ready, I can help you draft a clear, respectful incident report."
    )

def _ai_fallback_report(payload: dict) -> str:
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
    if not q:
        return jsonify({"answer": "Please enter a question."})

    # If no OpenAI, return fallback immediately
    if not client:
        return jsonify({"answer": _ai_fallback_answer(q)})

    # Try OpenAI with hard timeout; if it fails, send fallback
    try:
        # New OpenAI Python SDK (v1.x): responses API with a per-request timeout
        resp = client.responses.with_options(timeout=10).create(
            model="gpt-4o-mini",
            input=f"You are a supportive, non-judgmental safety assistant. "
                  f"Be concise, avoid stereotypes, and give practical steps. Question: {q}"
        )
        answer = resp.output_text or _ai_fallback_answer(q)
        return jsonify({"answer": answer})
    except Exception as e:
        # Don’t 500 the app — return a helpful fallback
        msg = _ai_fallback_answer(q) + "\n\n(Note: AI had an issue; fallback shown.)"
        return jsonify({"answer": msg})

@app.post("/generate_report")
def generate_report():
    payload = request.get_json(force=True) or {}

    # If no OpenAI, return a structured draft so the app doesn’t hang
    if not client:
        return jsonify({"report": _ai_fallback_report(payload)})

    prompt = textwrap.dedent(f"""
    Draft a clear, neutral incident report using these details.
    Include: who/what/when/where, impact, and next steps (if appropriate).
    Avoid stereotypes or assumptions.

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
        text = resp.output_text or _ai_fallback_report(payload)
        return jsonify({"report": text})
    except Exception:
        return jsonify({"report": _ai_fallback_report(payload)})
