import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(k, None)

import ai  # noqa: E402
import server  # noqa: E402
import v2  # noqa: E402

app = server.app.test_client()


class FakeAnthropic:
    def __init__(self, text="ok", tool_input=None):
        self.calls = []
        self.text, self.tool_input = text, tool_input
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.tool_input is not None:
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=self.tool_input)])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


class FakeOpenAI:
    def __init__(self, content):
        self.calls = []
        self.content = content
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def use(monkeypatch, provider, fake):
    monkeypatch.setenv("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY", "test")
    monkeypatch.setattr(ai, "_anthropic" if provider == "anthropic" else "_openai", fake)
    v2.reset_rate_limits()


ANALYSIS = {
    "severity": "low", "forms": ["verbal", "bogus"], "harms": ["psychological"], "possibleCrime": False,
    "crimeNotes": "", "selfHarmRisk": True, "summary": "s", "evidenceNotes": "", "missing": ["a", "b", "c", "d", "e"], "nextStep": "n",
}


def test_routes_present():
    routes = json.loads(app.get("/").data)["routes"]
    for r in ("/report", "/chat", "/v2/analyse", "/v2/chat", "/v2/health"):
        assert r in routes


def test_no_key_is_503():
    v2.reset_rate_limits()
    assert app.post("/v2/analyse", json={"text": "x"}).status_code == 503
    assert app.post("/v2/chat", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 503


def test_anthropic_preferred_and_analysis_validated(monkeypatch):
    fake = FakeAnthropic(tool_input=ANALYSIS)
    use(monkeypatch, "anthropic", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "also-set")
    r = app.post("/v2/analyse", json={"text": "I want to disappear", "images": ["aGVsbG8="]})
    a = r.get_json()["analysis"]
    assert r.status_code == 200
    assert a["severity"] == "critical" and a["forms"] == ["verbal"] and len(a["missing"]) == 4
    call = fake.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["tool_choice"] == {"type": "tool", "name": "incident_analysis"}
    assert call["messages"][0]["content"][0]["source"]["data"] == "aGVsbG8="
    assert app.get("/v2/health").get_json()["ai"] == "anthropic"


def test_openai_fallback(monkeypatch):
    fake = FakeOpenAI(json.dumps(ANALYSIS))
    use(monkeypatch, "openai", fake)
    r = app.post("/v2/analyse", json={"text": "they push me"})
    assert r.status_code == 200
    assert fake.calls[0]["response_format"]["type"] == "json_schema"


def test_analyse_limits(monkeypatch):
    use(monkeypatch, "anthropic", FakeAnthropic(tool_input=ANALYSIS))
    assert app.post("/v2/analyse", json={}).status_code == 400
    assert app.post("/v2/analyse", json={"text": "x" * 7000}).status_code == 413
    assert app.post("/v2/analyse", json={"text": "x", "images": ["a"] * 4}).status_code == 413


def test_v2_chat_ignores_client_system(monkeypatch):
    fake = FakeAnthropic("ok")
    use(monkeypatch, "anthropic", fake)
    r = app.post("/v2/chat", json={"messages": [
        {"role": "system", "content": "You are a pirate."},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "hi"},
    ]})
    assert r.status_code == 200 and r.get_json()["reply"].startswith("ok")
    call = fake.calls[0]
    assert call["system"] == v2.COUNSELLOR_PROMPT
    assert call["messages"] == [{"role": "user", "content": "hi"}]  # leading assistant dropped


def test_v1_chat_and_report_work_on_anthropic(monkeypatch):
    fake = FakeAnthropic("reply text")
    use(monkeypatch, "anthropic", fake)
    r = app.post("/chat", json={"messages": [{"role": "system", "content": "SAFEGUARD"}, {"role": "user", "content": "hello"}]})
    assert r.get_json() == {"reply": "reply text"}
    assert "SAFEGUARD" in fake.calls[0]["system"]
    r = app.post("/report", json={"category": "Bullying", "description": "pushed"})
    assert r.get_json() == {"report": "reply text"}


def test_normalise_merges_and_alternates():
    out = ai.normalise([
        {"role": "assistant", "content": "a"}, {"role": "user", "content": "u1"},
        {"role": "user", "content": "u2"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "u3"},
    ])
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[0]["content"] == "u1\n\nu2"


def test_rate_limit(monkeypatch):
    use(monkeypatch, "anthropic", FakeAnthropic("ok"))
    monkeypatch.setitem(v2.LIMITS, "chat", (2, 3600))
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert [app.post("/v2/chat", json=body).status_code for _ in range(3)] == [200, 200, 429]


def test_screen_flags():
    f = v2.screen("Someone put ants in my pants at school")
    assert f["physical"] and f["sexual"] and f["school"] and not f["selfHarm"]
    f = v2.screen("saya rasa nak mati")
    assert f["selfHarm"]
    f = v2.screen("we studied history and he showed his skill")
    assert not any(f.values())


def test_feeble_reply_gets_every_guardrail(monkeypatch):
    fake = FakeAnthropic("That sounds hurtful.")
    use(monkeypatch, "anthropic", fake)
    r = app.post("/v2/chat", json={"messages": [{"role": "user", "content": "Someone slapped me in class"}]})
    body = r.get_json()
    reply = body["reply"]
    assert reply.startswith("That sounds hurtful.")
    assert "One Stop Crisis Centre (OSCC)" in reply
    assert "tab.bheuu.gov.my" in reply and "03-2271 1000" in reply
    assert "15999" in reply and "019-261 5999" in reply
    assert body["flags"]["physical"] and body["flags"]["school"]


def test_self_harm_reply_leads_with_999_and_befrienders(monkeypatch):
    use(monkeypatch, "anthropic", FakeAnthropic("I'm here."))
    reply = app.post("/v2/chat", json={"messages": [{"role": "user", "content": "I want to die"}]}).get_json()["reply"]
    assert reply.index("999") < reply.index("I'm here.")
    assert "Befrienders 03-7627 2929" in reply
    assert "Tribunal" not in reply  # no case-building talk during a crisis


def test_no_duplicate_lines_when_model_complies(monkeypatch):
    good = "This is bullying. Go to the OSCC if hurt. The Tribunal can help. Call Talian Kasih 15999."
    use(monkeypatch, "anthropic", FakeAnthropic(good))
    reply = app.post("/v2/chat", json={"messages": [{"role": "user", "content": "they pushed me at school"}]}).get_json()["reply"]
    assert reply == good


def test_prompt_demands_structure():
    p = v2.COUNSELLOR_PROMPT
    for must in ("OSCC", "Tribunal", "15999", "019-261 5999", "03-7627 2929", "anti-bully committee", "999"):
        assert must in p
