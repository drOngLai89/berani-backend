import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Optional: use OpenAI if key present; otherwise fall back to a template generator
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = Flask(__name__)
CORS(app)

def template_report(category, dateISO, timeISO, locationText, description):
    dt = datetime.fromisoformat(dateISO.replace('Z', '+00:00'))
    tt = datetime.fromisoformat(timeISO.replace('Z', '+00:00'))
    date_str = dt.strftime('%A, %d %B %Y')
    time_str = tt.strftime('%I:%M %p')
    loc = locationText.strip() or '(Location not provided)'
    body = f"""Incident Report — {category}

Date: {date_str}
Time: {time_str}
Location: {loc}

Summary:
{description.strip()}

Observed Facts:
- Incident categorized as {category}.
- Occurred on {date_str} at approximately {time_str}.
- Location reported as: {loc}.
- Attached evidence: (see photos if provided in the app).

Impact & Next Steps:
- Recommend speaking with relevant authorities/teachers or counselors.
- Suggest preserving any additional evidence (messages, screenshots, witness statements).
- Consider follow-up within 48 hours to ensure safety and resolution.
"""
    return body

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.get_json(force=True)
    category = data.get('category', 'Bullying')
    dateISO = data.get('dateISO')
    timeISO = data.get('timeISO')
    locationText = data.get('locationText', '')
    description = data.get('description', '')

    if not dateISO or not timeISO or not description:
        return jsonify(error='Missing fields: dateISO, timeISO, description are required'), 400

    if USE_OPENAI:
        prompt = f"""
You are a school safeguarding officer. Write a concise, professional incident report for administration.
Inputs:
- Category: {category}
- Date: {dateISO}
- Time: {timeISO}
- Location: {locationText}
- Description: {description}

Requirements:
- Neutral, factual tone.
- Structure with headings: Incident Overview, Timeline, Location, Details, Evidence (if any), Recommended Actions.
- No speculation; only facts and reasonable recommendations.
"""
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2
            )
            text = completion.choices[0].message.content.strip()
            return jsonify(report=text)
        except Exception as e:
            # Fall back to template on any API error
            return jsonify(report=template_report(category, dateISO, timeISO, locationText, description))
    else:
        return jsonify(report=template_report(category, dateISO, timeISO, locationText, description))

@app.route('/assistant', methods=['POST'])
def assistant():
    data = request.get_json(force=True)
    q = data.get('question', '').strip()
    if not q:
        return jsonify(error='Missing question'), 400

    if USE_OPENAI:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"You are Berani Assistant. Keep answers brief, practical, and safe."},
                          {"role":"user","content":q}],
                temperature=0.3
            )
            text = completion.choices[0].message.content.strip()
            return jsonify(answer=text)
        except Exception:
            pass

    # Fallback canned helper
    answer = ("Keep reports factual and specific. Capture names (or descriptions), exact time/date, location, and what was said/done. "
              "Attach clear evidence (photos/screenshots). Use respectful language and avoid speculation.")
    return jsonify(answer=answer)

@app.route('/')
def root():
    return jsonify(ok=True)
