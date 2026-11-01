"""LLM wrappers. Default: Gemini via the current `google-genai` SDK (the old
`google-generativeai` package and gemini-2.0-flash are retired)."""
from __future__ import annotations
import json, os, re
from typing import Optional

from pydantic import BaseModel


class PaattiReply(BaseModel):
    intent: str            # advice | lookup | about | smalltalk | offtopic | crisis
    verse_no: Optional[int] = None
    reason: str = ""
    response: str


class LLM:
    name = "base"
    def reply(self, system: str, turns: list[dict]) -> PaattiReply:
        raise NotImplementedError
    def transcribe(self, audio: bytes, mime: str) -> str:
        raise NotImplementedError


class DailyQuotaExceeded(RuntimeError):
    """Free-tier per-day quota for a model is used up; retrying won't help until it resets."""


class GeminiLLM(LLM):
    """Gemini via google-genai. Default gemini-3.8-flash (gemini-2.0/2.5-flash are closed to new users).
    Gemini 3+: thinking is set with thinking_level (low|medium|high) and sampling params like
    temperature must not be sent. Older models still get temperature + thinking_budget."""
    def __init__(self, model: Optional[str] = None, temperature: float = 0.6):
        from google import genai
        from .retrieval import _api_key
        from google.genai import types
        # Turn off the SDK's silent internal retries; _send() retries visibly with a cap and timeout.
        self.client = genai.Client(api_key=_api_key(), http_options=types.HttpOptions(
            timeout=int(float(os.getenv("PAATTI_LLM_TIMEOUT", "60")) * 1000),
            retry_options=types.HttpRetryOptions(attempts=1)))
        # Default is Flash-Lite: on the free tier Flash models allow ~20 requests/day, Flash-Lite ~500/day.
        self.model = model or os.getenv("PAATTI_LLM", "gemini-3.5-flash-lite")
        self.fallback_model = os.getenv("PAATTI_LLM_FALLBACK") or None  # e.g. another Flash model id
        self.max_retries = int(os.getenv("PAATTI_LLM_RETRIES", "4"))
        self.verbose = True
        self.name = f"gemini:{self.model}"
        self.temperature = temperature
        m = re.search(r"gemini-(\d+)", self.model)
        self.gen = int(m.group(1)) if m else 3
        self.thinking_level = os.getenv("PAATTI_THINKING_LEVEL", "low")
        self._schema_ok = True  # flips off if the API rejects response_schema
        self._thinking_ok = True  # flips off if the model rejects thinking_config
        rpm = float(os.getenv("PAATTI_LLM_RPM", "0") or 0)  # client-side throttle; 0 = off
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0

    def _config(self, system=None, json_schema=None, json_mode=False):
        from google.genai import types
        kw = {}
        if system:
            kw["system_instruction"] = system
        if self.gen >= 3 and self._thinking_ok:
            kw["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        else:
            kw["temperature"] = self.temperature
            if "2.5" in self.model:
                kw["thinking_config"] = types.ThinkingConfig(thinking_budget=int(os.getenv("PAATTI_THINKING_BUDGET", "512")))
        if json_schema is not None or json_mode:
            kw["response_mime_type"] = "application/json"
        if json_schema is not None:
            kw["response_schema"] = json_schema
        return types.GenerateContentConfig(**kw)

    _TRANSIENT = (429, 500, 502, 503, 504)
    _last_call: dict = {}  # model -> monotonic time of last request (shared across instances)

    def _throttle(self, model):
        import time
        if self.min_interval:
            wait = GeminiLLM._last_call.get(model, 0) + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        GeminiLLM._last_call[model] = time.monotonic()

    @staticmethod
    def _is_daily_quota(err) -> bool:
        t = str(err)
        if "PerDay" in t or "per day" in t.lower():
            return True
        m = re.search(r"retryDelay'?:\s*'?([0-9.]+)s", t)
        return bool(m and float(m.group(1)) > 300)

    def _send(self, contents, config):
        """generate_content with visible retry/backoff on overload (503) and per-minute limits (429).
        A per-day quota stops immediately (DailyQuotaExceeded) or moves on to PAATTI_LLM_FALLBACK."""
        import time
        from google.genai import errors
        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        last = None
        for model in models:
            delay = 4.0
            attempt = 0
            while True:
                attempt += 1
                self._throttle(model)
                try:
                    return self.client.models.generate_content(model=model, contents=contents, config=config)
                except errors.APIError as e:
                    code = getattr(e, "code", None)
                    if code == 400 and "thinking" in str(e).lower() and getattr(config, "thinking_config", None):
                        self._thinking_ok = False  # model doesn't take thinking settings: resend without
                        config = config.model_copy(update={"thinking_config": None})
                        continue
                    if code not in self._TRANSIENT:
                        raise
                    last = e
                    if code == 429 and self._is_daily_quota(e):
                        last = DailyQuotaExceeded(
                            f"Daily free-tier quota used up for {model}. It resets at midnight Pacific time. "
                            f"Use another model (--llm gemini:<id>), set PAATTI_LLM_FALLBACK, or enable billing.")
                        break
                except Exception as e:  # timeouts / connection resets
                    if "timeout" not in type(e).__name__.lower() and "connect" not in type(e).__name__.lower():
                        raise
                    last, code = e, "timeout"
                if attempt > self.max_retries:
                    break
                m = re.search(r"retry(?:Delay| in)[^0-9]*([0-9.]+)\s*s", str(last))
                wait = min(float(m.group(1)) + 1, 90) if m else delay
                if self.verbose:
                    print(f"    Gemini {code} on {model}: retry {attempt}/{self.max_retries} in {wait:.0f}s", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 60)
            if self.verbose and self.fallback_model and model != models[-1]:
                print(f"    {model} unavailable, trying fallback {self.fallback_model}", flush=True)
        raise last

    def _call(self, contents, system=None, schema=None):
        """generate_content with graceful degradation: schema -> JSON mode -> plain text."""
        from google.genai import errors
        attempts = []
        if schema is not None and self._schema_ok:
            attempts.append(("schema", self._config(system, json_schema=schema)))
        if schema is not None:
            hint = ("\n\nReturn ONLY a JSON object with keys: " + ", ".join(schema.model_fields)) if hasattr(schema, "model_fields") else ""
            attempts.append(("json", self._config((system or "") + hint, json_mode=True)))
        attempts.append(("plain", self._config(system)))
        last = None
        for mode, cfg in attempts:
            try:
                return self._send(contents, cfg)
            except errors.ClientError as e:
                last = e
                if getattr(e, "code", None) == 404:
                    raise RuntimeError(f"Gemini model '{self.model}' is not available to this key ({e}). "
                                       "Set PAATTI_LLM in .env to a current model id.") from e
                if getattr(e, "code", None) != 400:
                    raise
                if mode == "schema":
                    self._schema_ok = False
        raise last

    def reply(self, system, turns):
        from google.genai import types
        contents = [types.Content(role="user" if t["role"] == "user" else "model",
                                  parts=[types.Part.from_text(text=t["content"])]) for t in turns]
        r = self._call(contents, system, schema=PaattiReply)
        if isinstance(getattr(r, "parsed", None), PaattiReply):
            return r.parsed
        return PaattiReply(**_loose_json(r.text))

    def generate_json(self, prompt: str) -> dict:
        r = self._call_json(prompt)
        return _loose_json(r.text)

    def _call_json(self, prompt):
        from google.genai import errors
        try:
            return self._send(prompt, self._config(json_mode=True))
        except errors.ClientError as e:
            if getattr(e, "code", None) != 400:
                raise
            return self._send(prompt, self._config())

    def transcribe(self, audio, mime):
        from google.genai import types
        r = self._send([types.Part.from_bytes(data=audio, mime_type=mime or "audio/wav"),
                        "Transcribe this speech exactly as spoken, in whatever language(s) it is in. Write each "
                        "language in its OWN script: Tamil in Tamil script (தமிழ்), Hindi in Devanagari, Malayalam, "
                        "Telugu, Kannada, Bengali, Urdu in theirs; use Latin letters only for English words. Never "
                        "romanize an Indian language. Return only the transcript, nothing else. If there is no "
                        "intelligible speech (silence, noise, music), return an empty string."],
                       self._config())
        from .voice import clean_transcript
        return clean_transcript(r.text)


class MockLLM(LLM):
    """Offline stand-in for tests: picks the first candidate verse."""
    name = "mock"
    def reply(self, system, turns):
        m = re.search(r"CANDIDATE VERSES.*?\[(\d+)\]", turns[-1]["content"], re.S)
        n = int(m.group(1)) if m else None
        return PaattiReply(intent="advice" if n else "smalltalk", verse_no=n, reason="mock",
                           response="(mock reply) Let this verse guide you.")
    def transcribe(self, audio, mime):
        return "mock transcript"


def _loose_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    return json.loads(m.group(0)) if m else {"intent": "advice", "response": text or ""}


def make_llm(spec: Optional[str] = None) -> LLM:
    spec = spec or os.getenv("PAATTI_LLM_PROVIDER", "gemini")
    if spec == "mock":
        return MockLLM()
    if spec.startswith("gemini"):
        return GeminiLLM(spec.split(":", 1)[1] if ":" in spec else None)
    raise ValueError(spec)
