import base64
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAB_EMAIL_FILING"):
    os.environ.pop(k, None)

import ai  # noqa: E402
import server  # noqa: E402
import v2  # noqa: E402
import tab  # noqa: E402

app = server.app.test_client()


class FakeTool:
    def __init__(self, tool_input):
        self.calls, self.tool_input, self.messages = [], tool_input, self

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=self.tool_input)])


class FakeAudio:
    def __init__(self, segments=None):
        self.calls = []
        self.segments = segments if segments is not None else [
            {"text": "semalam dua senior tampar saya kat asrama", "no_speech_prob": 0.02, "avg_logprob": -0.2},
        ]
        self.audio = SimpleNamespace(transcriptions=self)

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(text=" ".join(s["text"] for s in self.segments), segments=self.segments)


def setup(monkeypatch, tool_input=None, audio=None):
    v2.reset_rate_limits()
    if tool_input is not None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "t")
        monkeypatch.setattr(ai, "_anthropic", FakeTool(tool_input))
    if audio is not None:
        monkeypatch.setenv("OPENAI_API_KEY", "t")
        monkeypatch.setattr(ai, "_openai", audio)


def test_routes_and_config(monkeypatch):
    routes = json.loads(app.get("/").data)["routes"]
    for r in ("/v2/transcribe", "/v2/intake", "/v2/borang", "/v2/config"):
        assert r in routes
    c = app.get("/v2/config").get_json()
    assert c["emailFiling"] is False and c["tabWhatsApp"] is None
    assert c["tab"]["links"]["askUs"].endswith("/tanya-kami")
    monkeypatch.setenv("TAB_EMAIL_FILING", "true")
    assert app.get("/v2/config").get_json()["emailFiling"] is True


def test_transcribe(monkeypatch):
    fake = FakeAudio()
    setup(monkeypatch, audio=fake)
    b64 = base64.b64encode(b"\x00fake-audio").decode()
    r = app.post("/v2/transcribe", json={"audio": b64, "mime": "audio/m4a"})
    body = r.get_json()
    assert r.status_code == 200 and "tampar" in body["text"]
    assert body["flags"]["physical"] and body["flags"]["school"]
    name, data, mime = fake.calls[0]["file"]
    assert name == "voice.m4a" and data == b"\x00fake-audio" and mime == "audio/m4a"
    assert "prompt" not in fake.calls[0] and fake.calls[0]["response_format"] == "verbose_json"


def test_transcribe_drops_non_speech(monkeypatch):
    fake = FakeAudio([
        {"text": "Saya sering melihat kes buli di sekolah.", "no_speech_prob": 0.92, "avg_logprob": -0.4},
        {"text": "dia tolak saya", "no_speech_prob": 0.05, "avg_logprob": -0.3},
        {"text": "mumble", "no_speech_prob": 0.1, "avg_logprob": -1.8},
    ])
    setup(monkeypatch, audio=fake)
    body = app.post("/v2/transcribe", json={"audio": base64.b64encode(b"x").decode(), "mime": "audio/wav"}).get_json()
    assert body["text"] == "dia tolak saya"


def test_transcribe_limits(monkeypatch):
    setup(monkeypatch, audio=FakeAudio())
    assert app.post("/v2/transcribe", json={"audio": "abc", "mime": "video/mp4"}).status_code == 400
    assert app.post("/v2/transcribe", json={"audio": "!!notbase64!!"}).status_code == 400
    assert app.post("/v2/transcribe", json={"audio": "A" * 4_600_000}).status_code == 413


def test_transcribe_needs_openai(monkeypatch):
    v2.reset_rate_limits()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "t")
    assert app.post("/v2/transcribe", json={"audio": base64.b64encode(b"x").decode()}).status_code == 503


INTAKE = {
    "incidents": [
        {"date": "2026-10-23", "time": "21:30", "place": "asrama", "whatHappened": "Dua senior tampar saya.", "who": "two seniors",
         "witnesses": "", "atSchool": "maybe", "forms": ["physical", "bogus"], "harms": ["physical"], "evidence": ""},
        {"date": "2027-01-01", "time": "after school", "place": "", "whatHappened": "x", "who": "", "witnesses": "",
         "atSchool": "no", "forms": [], "harms": [], "evidence": ""},
    ],
    "unsure": ["date"], "questions": ["Siapa yang nampak?"], "severity": "high", "selfHarmRisk": False, "possibleCrime": True,
}


def test_intake_cleans_and_floors(monkeypatch):
    fake = FakeTool(INTAKE)
    setup(monkeypatch, tool_input=INTAKE)
    monkeypatch.setattr(ai, "_anthropic", fake)
    r = app.post("/v2/intake", json={"text": "semalam dua senior tampar saya kat asrama", "today": "2026-10-24"})
    body = r.get_json()
    a, b = body["incidents"]
    assert a["date"] == "2026-10-23" and a["time"] == "21:30" and a["atSchool"] == "unsure" and a["forms"] == ["physical"]
    assert b["date"] == "" and b["time"] == ""  # future date and non-time dropped, never guessed
    assert body["questions"] == ["Siapa yang nampak?"] and body["severity"] == "high"
    assert "TODAY: 2026-10-24" in fake.calls[0]["messages"][0]["content"][-1]["text"]


def test_intake_self_harm_is_critical(monkeypatch):
    setup(monkeypatch, tool_input={**INTAKE, "severity": "low"})
    body = app.post("/v2/intake", json={"text": "they laugh at me and I want to die"}).get_json()
    assert body["severity"] == "critical" and body["selfHarmRisk"] is True


def test_intake_never_loses_text(monkeypatch):
    setup(monkeypatch, tool_input={**INTAKE, "incidents": []})
    body = app.post("/v2/intake", json={"text": "they took my bag"}).get_json()
    assert body["incidents"][0]["whatHappened"] == "they took my bag"


def test_borang(monkeypatch):
    fake = FakeTool({"facts": "1. Pada 23 Oktober 2026 ...", "nature": "Buli fizikal"})
    setup(monkeypatch, tool_input={})
    monkeypatch.setattr(ai, "_anthropic", fake)
    r = app.post("/v2/borang", json={"language": "ms", "filer": "parent", "incidents": [{"date": "2026-10-23", "whatHappened": "ditampar"}]})
    assert r.get_json() == {"facts": "1. Pada 23 Oktober 2026 ...", "nature": ""}
    prompt = fake.calls[0]["messages"][0]["content"][-1]["text"]
    assert "Bahasa Melayu" in prompt and "parent" in prompt and "ditampar" in prompt
    assert app.post("/v2/borang", json={"incidents": []}).status_code == 400


class FakeChat:
    def __init__(self, text):
        self.text, self.messages, self.calls = text, self, []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


def test_official_questions_go_to_tanya_kami(monkeypatch):
    v2.reset_rate_limits()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "t")
    fake = FakeChat("I understand you want to know.")
    monkeypatch.setattr(ai, "_anthropic", fake)
    body = app.post("/v2/chat", json={"messages": [{"role": "user", "content": "Am I eligible for compensation from the tribunal?"}]}).get_json()
    assert body["flags"]["official"] is True
    assert "tab.bheuu.gov.my/tanya-kami" in body["reply"]
    system = fake.calls[0]["system"]
    assert "TRIBUNAL FACTS" in system and "Teachers cannot file" in system and "Never decide or predict" in system


def test_tab_facts_are_sourced():
    assert all(isinstance(f, str) and f for f in tab.TAB_FACTS)
    assert tab.LINKS["startComplaint"] == "https://tab.bheuu.gov.my/aduan/mula"


def test_questions_already_answered_are_dropped(monkeypatch):
    data = {**INTAKE, "incidents": [{**INTAKE["incidents"][0], "atSchool": "yes", "witnesses": "Daniel"}],
            "questions": ["Bila tepatnya kejadian itu berlaku?", "Di mana kejadian itu berlaku?", "Siapa lagi yang nampak?", "Ada simpan tangkapan skrin?"]}
    setup(monkeypatch, tool_input=data)
    body = app.post("/v2/intake", json={"text": "semalam senior tampar saya kat asrama", "today": "2026-10-24"}).get_json()
    assert body["questions"] == ["Ada simpan tangkapan skrin?"]


def test_borang_nature_and_paragraphs_are_deterministic(monkeypatch):
    fake = FakeTool({"facts": "1. Pada 23 Okt di kantin. 2. Pada 24 Okt di WhatsApp.", "nature": "physical, online"})
    setup(monkeypatch, tool_input={})
    monkeypatch.setattr(ai, "_anthropic", fake)
    body = app.post("/v2/borang", json={"language": "ms", "incidents": [
        {"date": "2026-10-23", "whatHappened": "a", "forms": ["online", "physical"]}]}).get_json()
    assert body["nature"] == "Buli fizikal, siber"
    assert body["facts"] == "1. Pada 23 Okt di kantin.\n\n2. Pada 24 Okt di WhatsApp."


class LangAudio:
    """Whisper that guesses Malay for English speech and writes a (low confidence) translation."""
    def __init__(self):
        self.calls = []
        self.audio = SimpleNamespace(transcriptions=self)

    def create(self, **kw):
        self.calls.append(kw)
        lang = kw.get("language")
        if lang == "en":
            segs = [{"text": "Yesterday two seniors pushed me in the canteen.", "no_speech_prob": 0.02, "avg_logprob": -0.15}]
            return SimpleNamespace(language="english", segments=segs)
        segs = [{"text": "Semalam dua senior menolak saya di kantin.", "no_speech_prob": 0.02, "avg_logprob": -0.85}]
        return SimpleNamespace(language="malay", segments=segs)


def test_transcribe_keeps_the_language_spoken(monkeypatch):
    fake = LangAudio()
    setup(monkeypatch, audio=fake)
    body = app.post("/v2/transcribe", json={"audio": base64.b64encode(b"x").decode(), "mime": "audio/m4a"}).get_json()
    assert body["text"].startswith("Yesterday two seniors")
    assert [c.get("language") for c in fake.calls] == [None, "en"]


def test_transcribe_keeps_malay_when_it_is_malay(monkeypatch):
    class Malay(LangAudio):
        def create(self, **kw):
            self.calls.append(kw)
            if kw.get("language") == "en":
                return SimpleNamespace(language="english", segments=[{"text": "Yesterday two seniors pushed me.", "no_speech_prob": 0.02, "avg_logprob": -0.9}])
            return SimpleNamespace(language="malay", segments=[{"text": "Semalam dua senior tolak saya.", "no_speech_prob": 0.02, "avg_logprob": -0.2}])
    fake = Malay()
    setup(monkeypatch, audio=fake)
    body = app.post("/v2/transcribe", json={"audio": base64.b64encode(b"x").decode(), "mime": "audio/m4a"}).get_json()
    assert body["text"] == "Semalam dua senior tolak saya."


def test_transcribe_forced_language(monkeypatch):
    fake = LangAudio()
    setup(monkeypatch, audio=fake)
    app.post("/v2/transcribe", json={"audio": base64.b64encode(b"x").decode(), "mime": "audio/m4a", "language": "en"})
    assert [c.get("language") for c in fake.calls] == ["en"]
