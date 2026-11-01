"""Offline tests (no API keys needed): python -m pytest -q"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from paatti import KnowledgeBase, Paatti
from paatti.llm import LLM, MockLLM, PaattiReply
from paatti.safety import is_crisis

KB = KnowledgeBase()


def test_kb_complete():
    assert len(KB) == 109 and [v.no for v in KB.verses] == list(range(1, 110))
    assert all(v.situations and v.meaning for v in KB.verses)


@pytest.mark.parametrize("q,n", [
    ("What does verse 54 mean?", 54), ("Explain ஆறுவது சினம்", 2), ("meaning of nandri maravel", 21),
    ("tell me the 100th aathichoodi", 100), ("what is the last verse", 109), ("ஓதுவது ஒழியேல் meaning", 11),
])
def test_lookup(q, n):
    assert KB.lookup(q) == n


@pytest.mark.parametrize("q", ["I get angry easily", "hi paatti", "my friend cheated me", "I lost 5 kg"])
def test_no_false_lookup(q):
    assert KB.lookup(q) is None


def test_crisis():
    assert is_crisis("I don't want to live anymore") and is_crisis("thinking of suicide")
    assert not is_crisis("I want to live a better life")


class BadLLM(LLM):
    """Returns a hallucinated verse number to check validation."""
    name = "bad"
    def reply(self, system, turns):
        return PaattiReply(intent="advice", verse_no=999, response="Kanna...")


class OfftopicLLM(LLM):
    name = "off"
    def reply(self, system, turns):
        return PaattiReply(intent="offtopic", verse_no=5, response="Kanna, Paatti only knows life wisdom.")


def test_pipeline_mock():
    bot = Paatti(llm=MockLLM(), embedder_spec="none")
    o = bot.answer("I keep procrastinating and postponing my work")
    assert o["verse"]["no"] == o["verse_no"] and o["verse"]["tamil"] == KB.get(o["verse_no"]).tamil
    assert o["verse_no"] in o["candidates"]


def test_crisis_short_circuits():
    o = Paatti(llm=BadLLM(), embedder_spec="none").answer("I want to die")
    assert o["intent"] == "crisis" and o["verse"] is None and "14416" in o["response"]


def test_hallucinated_verse_is_replaced():
    o = Paatti(llm=BadLLM(), embedder_spec="none").answer("I get angry very easily")
    assert KB.get(o["verse_no"]) is not None


def test_offtopic_has_no_verse():
    o = Paatti(llm=OfftopicLLM(), embedder_spec="none").answer("write bubble sort in python")
    assert o["verse"] is None


def test_lookup_forces_verse():
    o = Paatti(llm=OfftopicLLM(), embedder_spec="none").answer("What does verse 54 mean?")
    assert o["verse_no"] == 54 and o["intent"] == "lookup"


def test_llm_failure_falls_back():
    class Boom(LLM):
        def reply(self, s, t): raise RuntimeError("quota")
    o = Paatti(llm=Boom(), embedder_spec="none").answer("I am lazy")
    assert o["intent"] == "fallback" and o["verse"]


def test_api():
    from fastapi.testclient import TestClient
    import os
    os.environ["PAATTI_LLM_PROVIDER"] = "mock"; os.environ["PAATTI_EMBEDDER"] = "none"
    import api
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/chat", json={"query": "I spend all my salary", "history": [
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello kanna", "verse_no": None}]})
        assert r.status_code == 200 and r.json()["verse"]
        assert c.get("/verse/2").json()["tamil"] == "ஆறுவது சினம்"
        assert c.get("/verse/500").status_code == 404


def test_registry_prompts_applied(monkeypatch):
    """STEmbedder must prepend the registry's query/doc prompts (no network: fake SentenceTransformer)."""
    import types, numpy as np
    seen = {}
    class FakeST:
        def __init__(self, mid, **kw): seen["id"], seen["remote"] = mid, kw.get("trust_remote_code")
        def get_sentence_embedding_dimension(self): return 4
        def encode(self, texts, **kw):
            seen.setdefault("texts", []).extend(texts); return np.ones((len(texts), 4))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    from paatti.retrieval import make_embedder
    e = make_embedder("e5-large"); e.encode(["hi"], True); e.encode(["doc"], False)
    assert seen["id"] == "intfloat/multilingual-e5-large" and seen["texts"] == ["query: hi", "passage: doc"]
    seen.clear(); e = make_embedder("qwen3-0.6b"); e.encode(["x"], True); e.encode(["y"], False)
    assert seen["texts"][0].startswith("Instruct: ") and seen["texts"][1] == "y"
    seen.clear(); make_embedder("gte-ml"); assert seen["remote"] is True


def test_no_finetuned_models_in_registry():
    from paatti.embedders import MODELS
    assert not any(m["id"].startswith("GSR-608001/") for m in MODELS.values())


class _FakeEmb:
    """Deterministic bag-of-words embedder for offline tests."""
    name = "fake"
    def encode(self, texts, is_query):
        import hashlib, numpy as np
        from paatti.retrieval import tokenize
        v = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tokenize(t) or ["_empty_"]:
                v[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_dense_only_by_default(tmp_path):
    from paatti.retrieval import HybridRetriever
    r = HybridRetriever(KB, _FakeEmb(), cache_dir=tmp_path)
    assert r.bm25 is None and r.bm25_weight == 0
    r2 = HybridRetriever(KB, _FakeEmb(), cache_dir=tmp_path, bm25_weight=0.3)
    assert r2.bm25 is not None and r2.search("I get angry", k=3)


def test_bm25_skipped_when_no_keyword_overlap(tmp_path):
    """Tamil-script query: BM25 scores are all zero, so fusion must equal dense-only ranking."""
    from paatti.retrieval import HybridRetriever
    q = "என் நண்பன் என்னை ஏமாற்றிவிட்டான்"
    dense = [n for n, _ in HybridRetriever(KB, _FakeEmb(), cache_dir=tmp_path, use_bm25=False).search(q, k=10)]
    fused = [n for n, _ in HybridRetriever(KB, _FakeEmb(), cache_dir=tmp_path, bm25_weight=1.0).search(q, k=10)]
    assert dense == fused


def test_keyword_only_without_embedder():
    from paatti.retrieval import HybridRetriever
    r = HybridRetriever(KB, None)
    assert r.search("I spend my whole salary on shopping", k=5)
    assert r.search("என் நண்பன்", k=5)  # no overlap still returns something


def test_api_key_read_from_dotenv(tmp_path, monkeypatch):
    import paatti.retrieval as R
    for k in R._KEY_NAMES:
        monkeypatch.delenv(k, raising=False)
    (tmp_path / ".env").write_text('GOOGLE_API_KEY = "abc123"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(R, "__file__", str(tmp_path / "x" / "paatti" / "retrieval.py"))  # project .env absent
    assert R._api_key() == "abc123"


def _fake_gemini(monkeypatch, behaviour):
    """GeminiLLM with a fake client. behaviour(config) -> response or raises."""
    from types import SimpleNamespace
    from google.genai import errors
    import paatti.llm as L
    calls = []
    class Models:
        def generate_content(self, model, contents, config):
            calls.append(config)
            return behaviour(config, errors, SimpleNamespace)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kw: SimpleNamespace(models=Models()))
    monkeypatch.setattr("time.sleep", lambda s: None)
    return L.GeminiLLM(), calls


def test_gemini3_config_no_temperature(monkeypatch):
    pytest.importorskip("google.genai")
    def ok(cfg, errors, NS):
        return NS(parsed=PaattiReply(intent="advice", verse_no=2, response="Kanna"), text="")
    llm, calls = _fake_gemini(monkeypatch, ok)
    assert llm.model == "gemini-3.5-flash-lite"
    r = llm.reply("sys", [{"role": "user", "content": "angry"}])
    assert r.verse_no == 2
    cfg = calls[0]
    assert cfg.temperature is None and cfg.thinking_config.thinking_level is not None
    assert cfg.thinking_config.thinking_budget is None and cfg.response_schema is not None


def test_schema_rejected_falls_back_to_json(monkeypatch):
    pytest.importorskip("google.genai")
    def beh(cfg, errors, NS):
        if cfg.response_schema is not None:
            raise errors.ClientError(400, {"error": {"code": 400, "message": "schema unsupported", "status": "INVALID_ARGUMENT"}})
        return NS(parsed=None, text='{"intent":"advice","verse_no":54,"reason":"r","response":"Kanna"}')
    llm, calls = _fake_gemini(monkeypatch, beh)
    assert llm.reply("sys", [{"role": "user", "content": "lazy"}]).verse_no == 54
    assert llm._schema_ok is False and calls[1].response_mime_type == "application/json"


def test_model_404_gives_clear_error(monkeypatch):
    pytest.importorskip("google.genai")
    def beh(cfg, errors, NS):
        raise errors.ClientError(404, {"error": {"code": 404, "message": "no longer available", "status": "NOT_FOUND"}})
    llm, _ = _fake_gemini(monkeypatch, beh)
    with pytest.raises(RuntimeError, match="PAATTI_LLM"):
        llm.reply("sys", [{"role": "user", "content": "x"}])


def test_503_is_retried_then_succeeds(monkeypatch):
    pytest.importorskip("google.genai")
    n = {"c": 0}
    def beh(cfg, errors, NS):
        n["c"] += 1
        if n["c"] < 3:
            raise errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})
        return NS(parsed=PaattiReply(intent="advice", verse_no=6, response="Kanna"), text="")
    llm, calls = _fake_gemini(monkeypatch, beh)
    assert llm.reply("s", [{"role": "user", "content": "x"}]).verse_no == 6 and n["c"] == 3


def test_503_exhausted_uses_fallback_model(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("PAATTI_LLM_FALLBACK", "gemini-backup")
    monkeypatch.setenv("PAATTI_LLM_RETRIES", "1")
    from types import SimpleNamespace
    from google.genai import errors
    import paatti.llm as L
    used = []
    class Models:
        def generate_content(self, model, contents, config):
            used.append(model)
            if model != "gemini-backup":
                raise errors.ServerError(503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}})
            return SimpleNamespace(parsed=PaattiReply(intent="advice", verse_no=2, response="k"), text="")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kw: SimpleNamespace(models=Models()))
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert L.GeminiLLM().reply("s", [{"role": "user", "content": "x"}]).verse_no == 2
    assert used == ["gemini-3.5-flash-lite", "gemini-3.5-flash-lite", "gemini-backup"]


def test_eval_resume_retries_failed_rows(tmp_path, monkeypatch):
    import json as _j, argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    import run_eval as R
    monkeypatch.setattr(R, "EVAL", R.EVAL[:3])
    out = tmp_path / "res.jsonl"
    # row 0 done, row 1 failed earlier -> should be rerun along with row 2
    out.write_text(_j.dumps({"id": 0, "intent": "advice", "verse_no": 2, "type": "advice", "correct": True}) + "\n"
                   + _j.dumps({"id": 1, "intent": "fallback", "verse_no": 5, "type": "advice", "correct": False}) + "\n")
    a = argparse.Namespace(llm="mock", embedder="none", judge=False, judge_llm="same", rpm=0, limit=0, resume=str(out), sleep=0)
    R.cmd_pipeline(a)
    lines = [_j.loads(l) for l in out.read_text().splitlines()]
    assert [l["id"] for l in lines] == [0, 1, 1, 2]


_DAILY = {"error": {"code": 429, "message": "Quota exceeded ... requests per day ... Please retry in 12h16m31s.",
                    "status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "44191s"}]}}


def test_daily_quota_stops_without_retrying(monkeypatch):
    pytest.importorskip("google.genai")
    from paatti.llm import DailyQuotaExceeded
    n = {"c": 0}
    def beh(cfg, errors, NS):
        n["c"] += 1
        raise errors.ClientError(429, _DAILY)
    llm, _ = _fake_gemini(monkeypatch, beh)
    with pytest.raises(DailyQuotaExceeded):
        llm.reply("s", [{"role": "user", "content": "x"}])
    assert n["c"] == 1


def test_thinking_rejected_is_dropped(monkeypatch):
    pytest.importorskip("google.genai")
    def beh(cfg, errors, NS):
        if cfg.thinking_config is not None:
            raise errors.ClientError(400, {"error": {"code": 400, "message": "thinking_level is not supported", "status": "INVALID_ARGUMENT"}})
        return NS(parsed=PaattiReply(intent="advice", verse_no=7, response="k"), text="")
    llm, calls = _fake_gemini(monkeypatch, beh)
    assert llm.reply("s", [{"role": "user", "content": "x"}]).verse_no == 7 and llm._thinking_ok is False


def test_eval_stops_cleanly_on_daily_quota(tmp_path, monkeypatch, capsys):
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    import run_eval as R
    from paatti.llm import MockLLM
    class QuotaAfterOne(MockLLM):
        n = 0
        def reply(self, system, turns):
            QuotaAfterOne.n += 1
            if QuotaAfterOne.n > 2:  # preflight + first query succeed
                from paatti.llm import DailyQuotaExceeded
                raise DailyQuotaExceeded("Daily free-tier quota used up for m.")
            return super().reply(system, turns)
    monkeypatch.setattr("paatti.llm.make_llm", lambda spec=None: QuotaAfterOne())
    monkeypatch.setattr(R, "EVAL", R.EVAL[:5])
    out = tmp_path / "r.jsonl"
    R.cmd_pipeline(argparse.Namespace(llm="mock", embedder="none", judge=False, judge_llm="same", rpm=0,
                                      limit=0, resume=str(out), sleep=0))
    assert len(out.read_text().splitlines()) == 1 and "--resume" in capsys.readouterr().out


def test_throttle_spaces_requests(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("PAATTI_LLM_RPM", "6")  # 10 s apart
    slept = []
    def ok(cfg, errors, NS):
        return NS(parsed=PaattiReply(intent="advice", verse_no=1, response="k"), text="")
    llm, _ = _fake_gemini(monkeypatch, ok)
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    llm._last_call.clear()
    llm.reply("s", [{"role": "user", "content": "a"}]); llm.reply("s", [{"role": "user", "content": "b"}])
    assert slept and 9 < slept[0] <= 10


def test_rejudge_fills_missing_scores(tmp_path, monkeypatch):
    import json as _j, argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    import run_eval as R
    class J:
        name = "fakejudge"
        def generate_json(self, prompt):
            return {"relevance": 5, "faithfulness": 4, "warmth": 5, "actionability": 3}
    monkeypatch.setattr("paatti.llm.make_llm", lambda spec=None: J())
    v = KB.get(2).card()
    rows = [{"id": 0, "type": "advice", "intent": "advice", "verse": v, "verse_no": 2, "correct": True, "query": "q", "response": "r", "judge": None},
            {"id": 1, "type": "advice", "intent": "advice", "verse": v, "verse_no": 2, "correct": True, "query": "q", "response": "r",
             "judge": {"relevance": 1, "faithfulness": 1, "warmth": 1, "actionability": 1}}]
    f = tmp_path / "r.jsonl"; f.write_text("".join(_j.dumps(r) + "\n" for r in rows))
    R.cmd_rejudge(argparse.Namespace(file=str(f), judge_llm="x", rpm=0))
    out = [_j.loads(l) for l in f.read_text().splitlines()]
    assert out[0]["judge"]["faithfulness"] == 4 and out[1]["judge"]["relevance"] == 1


class _FixedLLM(LLM):
    """Returns a preset intent/verse to test the engine's verse-card rules."""
    name = "fixed"
    def __init__(self, intent, verse_no=None):
        self.intent, self.verse_no = intent, verse_no
    def reply(self, system, turns):
        return PaattiReply(intent=self.intent, verse_no=self.verse_no, response="Kanna...")


def test_about_question_shows_no_verse_card():
    """'How many verses are in Aathichoodi?' after a lookup must not re-show the previous verse."""
    hist = [{"role": "user", "content": "What does verse 54 mean?"},
            {"role": "assistant", "content": "...", "verse_no": 54}]
    o = Paatti(llm=_FixedLLM("about"), embedder_spec="none").answer("how many verses are in aathichoodi", hist)
    assert o["intent"] == "about" and o["verse"] is None


def test_about_with_example_verse_keeps_it():
    o = Paatti(llm=_FixedLLM("about", 1), embedder_spec="none").answer("what is the first verse about?")
    assert o["verse_no"] == 1


def test_advice_without_verse_falls_back_to_retrieval():
    o = Paatti(llm=_FixedLLM("advice", None), embedder_spec="none").answer("I keep shouting at my family in anger")
    assert o["verse"] is not None and o["verse_no"] == o["candidates"][0]


def test_unforced_lookup_with_invalid_verse_shows_no_card():
    o = Paatti(llm=_FixedLLM("lookup", 500), embedder_spec="none").answer("tell me about a verse on kindness")
    assert o["verse"] is None


@pytest.mark.parametrize("raw,expected", [
    ("En chellame, verse 54 is Sombith thiriyel.", "Verse 54 is Sombith thiriyel."),
    ("Kanna, it hurts when nobody listens.", "It hurts when nobody listens."),
    ("en chellame! Making plans is easy.", "Making plans is easy."),
    ("Finish one task today. Stay active and bright, my dear!", "Finish one task today. Stay active and bright!"),
    ("You are never truly alone, my child. 🌸", "You are never truly alone. 🌸"),
    ("Thangame, take a breath. Paatti is here, kanna!", "Take a breath. Paatti is here!"),
    ("The verse means be kind to the dear ones around you.", "The verse means be kind to the dear ones around you."),
    ("Krishna and Kannagi are names, not pet names.", "Krishna and Kannagi are names, not pet names."),
])
def test_strip_endearments(raw, expected):
    from paatti.engine import strip_endearments
    assert strip_endearments(raw) == expected


def test_no_pet_names_in_fixed_texts():
    from paatti.safety import CRISIS_RESPONSE
    from paatti.engine import SYSTEM
    assert "chellame" not in CRISIS_RESPONSE.lower() and "kanna" not in CRISIS_RESPONSE.lower()
    assert "Do NOT use pet names" in SYSTEM


def test_prompt_guards_unloved_users():
    from paatti.engine import SYSTEM
    assert "feeling unloved" in SYSTEM and "duties to parents" in SYSTEM
    assert "I feel unloved and like I don't belong anywhere" in KnowledgeBase().get(104).situations


def test_any_language_prompt_and_script_check():
    from paatti.engine import SYSTEM
    assert "any language" in SYSTEM and "Hinglish" in SYSTEM
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    from run_eval import script_ok
    assert script_ok("गुस्सा आने पर रुकिए और गहरी साँस लीजिए।", "devanagari")
    assert not script_ok("When anger comes, pause and breathe.", "devanagari")
    assert script_ok("ദേഷ്യം വരുമ്പോൾ ഒന്നു നിൽക്കൂ.", "malayalam")
    assert script_ok("Gussa aaye toh thoda ruko, Aaruvadhu sinam yaad karo.", "latin")


@pytest.mark.parametrize("raw,expected", [
    ("Hamesha sachai ke raste par chalte rahein mere bache.", "Hamesha sachai ke raste par chalte rahein."),
    ("Beta, gussa mat karo.", "Gussa mat karo."),
    ("हमेशा सच के रास्ते पर चलिए, मेरे बच्चे।", "हमेशा सच के रास्ते पर चलिए।"),
    ("The beta version is out.", "The beta version is out."),
    ("Use the beta", "Use the beta"),
    ("Mone, take care.", "Take care."),
])
def test_strip_endearments_other_languages(raw, expected):
    from paatti.engine import strip_endearments
    assert strip_endearments(raw) == expected


def test_named_language_uses_native_script():
    from paatti.engine import SYSTEM
    assert "ITS OWN SCRIPT" in SYSTEM and "Devanagari" in SYSTEM


def test_language_follows_current_message():
    from paatti.engine import SYSTEM
    assert "CURRENT message only" in SYSTEM and "from now on" in SYSTEM


@pytest.mark.parametrize("text,lang", [
    ("கோபம் வந்தால் சற்று நில்லுங்கள்.", "ta"), ("गुस्सा आए तो रुकिए।", "hi"), ("ദേഷ്യം വരുമ്പോൾ നിൽക്കൂ.", "ml"),
    ("When anger comes, pause.", "en"), ("Gussa aaye toh ruko, Aaruvadhu sinam yaad karo.", "en"),
    ("Aaruvadhu sinam என்பது கோபத்தை தணிக்க வேண்டும் என்பதாகும்.", "ta"), ("", "en"),
])
def test_tts_language(text, lang):
    from paatti.voice import tts_lang
    assert tts_lang(text) == lang


def test_transcribe_endpoint_and_prompt(monkeypatch):
    from fastapi.testclient import TestClient
    import os
    os.environ["PAATTI_LLM_PROVIDER"] = "mock"; os.environ["PAATTI_EMBEDDER"] = "none"
    import api
    with TestClient(api.app) as c:
        r = c.post("/transcribe", files={"file": ("a.wav", b"RIFF0000WAVE", "audio/wav")})
        assert r.status_code == 200 and r.json()["text"] == "mock transcript"
    import inspect
    from paatti.llm import GeminiLLM
    src = inspect.getsource(GeminiLLM.transcribe)
    assert "any language" in src and "empty string" in src


@pytest.mark.parametrize("raw,expected", [
    ("00:00", ""), ("...", ""), ("[silence]", ""), ("(no speech)", ""), ("No speech detected.", ""), ('""', ""),
    ("  ", ""), (None, ""),
    ("I get angry very easily. What should I do?", "I get angry very easily. What should I do?"),
    ("मैं अपनी सारी सैलरी खर्च कर देता हूं", "मैं अपनी सारी सैलरी खर्च कर देता हूं"),
    ("2 years ago my friend left", "2 years ago my friend left"),
])
def test_clean_transcript(raw, expected):
    from paatti.voice import clean_transcript
    assert clean_transcript(raw) == expected


def test_welcome_child_stripped():
    from paatti.engine import strip_endearments
    assert strip_endearments("Welcome, child. Tell me what is on your heart.") == "Welcome. Tell me what is on your heart."
    assert strip_endearments("Every child deserves care.") == "Every child deserves care."
    assert strip_endearments("Child, take a breath.") == "Take a breath."
