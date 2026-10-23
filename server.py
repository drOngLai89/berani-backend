from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import ai

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024
CORS(app)

from v2 import bp as v2_bp  # noqa: E402
app.register_blueprint(v2_bp)


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
    if ai.provider() is None:
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
        text = ai.chat("You write clear, supportive incident reports.", [{"role": "user", "content": prompt}], max_tokens=600)
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
    if ai.provider() is None:
        return jsonify({"reply": "I'm here with you. If you’re in danger call 999. 24/7 help: Talian Kasih 15999 (WhatsApp 019-2615999), Befrienders 03-7627 2929, WAO +603-7956 3488."})
    try:
        # v1 (Berani 2) clients send their own system prompt as a message; fold it into the system slot.
        sys_extra = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        convo = [{"role": m.get("role"), "content": str(m.get("content", ""))[:4000]}
                 for m in messages[-30:] if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()]
        if not convo or convo[-1]["role"] != "user":
            return jsonify({"error": "chat_failed"}), 400
        reply = ai.chat((system + "\n\n" + sys_extra).strip(), convo, max_tokens=700)
        return jsonify({"reply": reply})
    except Exception as e:
        print("chat error:", e)
        return jsonify({"error":"chat_failed"}), 500

for path in ("/chat", "/api/chat", "/v1/chat", "/messages", "/api/messages", "/respond"):
    ep = "chat_" + path.strip("/").replace("/", "_")
    app.add_url_rule(path, endpoint=ep, view_func=chat_handler, methods=["POST", "OPTIONS"])
