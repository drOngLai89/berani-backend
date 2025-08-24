from flask import Flask, request, jsonify
from flask_cors import CORS
import os
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

@app.get("/")
def health():
    routes = sorted({r.rule for r in app.url_map.iter_rules() if "static" not in r.rule})
    return jsonify({"ok": True, "routes": routes})

def report_handler():
    data = request.get_json(silent=True) or {}
    category = data.get("category") or "N/A"
    dateISO = data.get("dateISO") or "N/A"
    timeISO = data.get("timeISO") or "N/A"
    locationText = data.get("locationText") or "N/A"
    description = data.get("description") or ""
    if client is None:
        text = "### Description of the Incident:\n" + (description or "No description provided.") + \
               "\n\n### Impact:\nStill gathering details.\n\n### Next Steps:\nKeep evidence safely. In danger? Call 999 (Malaysia)."
        return jsonify({"report": text})
    try:
        prompt = f"""Write a clear, empathetic incident report for an app called Berani.

Context:
- Category: {category}
- Date: {dateISO}
- Time: {timeISO}
- Location: {locationText}

User description:
{description}

Requirements:
- Headings: Description of the Incident, Impact, Next Steps.
- Supportive tone, simple language, no PII, under 220 words."""
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
        text = resp.choices[0].message.content.strip()
        return jsonify({"report": text})
    except Exception as e:
        print("report error:", e)
        return jsonify({"error":"report_failed"}), 500

for path in ("/report", "/api/report", "/v1/report", "/generate", "/api/generate"):
    ep = "report_" + path.strip("/").replace("/", "_")
    app.add_url_rule(path, endpoint=ep, view_func=report_handler, methods=["POST", "OPTIONS"])

def chat_handler():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    system = data.get("system") or (
        "You are a compassionate counsellor for users in Malaysia. Ensure safety first: "
        "if risk of harm, advise calling 999 immediately. Provide practical next steps and local resources: "
        "Talian Kasih 15999 (WhatsApp 019-2615999), Befrienders 03-7627 2929, WAO +603-7956 3488 / WhatsApp +6018-988 8058. "
        "Avoid diagnosis; be brief, clear, and supportive."
    )
    if client is None:
        return jsonify({"reply": "I'm here with you. If you’re in danger call 999. 24/7 help: Talian Kasih 15999 (WhatsApp 019-2615999), Befrienders 03-7627 2929, WAO +603-7956 3488."})
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":system}] + [{"role":m.get("role","user"),"content":m.get("content","")} for m in messages])
        reply = resp.choices[0].message.content.strip()
        return jsonify({"reply": reply})
    except Exception as e:
        print("chat error:", e)
        return jsonify({"error":"chat_failed"}), 500

for path in ("/chat", "/api/chat", "/v1/chat", "/messages", "/api/messages", "/respond"):
    ep = "chat_" + path.strip("/").replace("/", "_")
    app.add_url_rule(path, endpoint=ep, view_func=chat_handler, methods=["POST", "OPTIONS"])
