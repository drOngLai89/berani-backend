set -e
BASE="${1:-https://berani-backend.onrender.com}"
curl -s "$BASE/" | grep '"ok": true' >/dev/null
curl -s -X POST "$BASE/report" -H "Content-Type: application/json" -d '{"category":"Test","dateISO":"2025-08-24","timeISO":"2025-08-24T11:00:00Z","locationText":"PJ","description":"hi"}' | grep '"report"' >/dev/null
curl -s -X POST "$BASE/chat" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hello"}]}' | grep '"reply"' >/dev/null
echo OK
