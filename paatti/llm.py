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


# Default pool, cheapest/highest-quota first. Free tier (Nov 2026): Flash-Lite ~500 requests/day each,
# Flash ~20/day each; every model has its own quota, so the pool's daily capacity is their sum.
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash",
                  "gemini-3.6-flash"]


def _next_pacific_midnight() -> float:
    """Epoch seconds of the next midnight in US Pacific time (when Gemini daily quotas reset)."""
    import datetime as dt
    from zoneinfo import ZoneInfo
    now = dt.datetime.now(ZoneInfo("America/Los_Angeles"))
    nxt = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
    return nxt.timestamp()


class GeminiLLM(LLM):
    """Gemini via google-genai with a POOL of models.

    strategy "round_robin" (default): each request starts at the next model in the pool, spreading load
    across separate free-tier quotas. "failover": always try the first available model, others only on
    failure (more consistent answers). Either way a model that is out of daily quota cools down until the
    Pacific-midnight reset, an overloaded/rate-limited one for a minute, an unavailable (404) one for a day,
    and the request moves on to the next model.

    Gemini 3+: thinking via thinking_level and no sampling params; older models get temperature/budget.
    """
    _lock = None
    _cooldown: dict = {}     # model -> epoch seconds until which it is skipped (shared by all instances)
    _rr = 0                  # round-robin cursor (shared)
    _last_call: dict = {}    # model -> monotonic time of last request (client-side throttle)
    _TRANSIENT = (429, 500, 502, 503, 504)

    def __init__(self, model: Optional[str] = None, temperature: float = 0.6):
        import threading
        from google import genai
        from google.genai import types
        from .retrieval import _api_key
        if GeminiLLM._lock is None:
            GeminiLLM._lock = threading.Lock()
        # Turn off the SDK's silent internal retries; _send() retries visibly with a cap and timeout.
        self.client = genai.Client(api_key=_api_key(), http_options=types.HttpOptions(
            timeout=int(float(os.getenv("PAATTI_LLM_TIMEOUT", "60")) * 1000),
            retry_options=types.HttpRetryOptions(attempts=1)))
        spec = model or os.getenv("PAATTI_LLM_MODELS") or os.getenv("PAATTI_LLM") or ",".join(DEFAULT_MODELS)
        self.models = [m.strip() for m in spec.split(",") if m.strip()]
        fb = os.getenv("PAATTI_LLM_FALLBACK")  # backward compatible: appended to the pool
        if fb and fb not in self.models:
            self.models.append(fb)
        self.model = self.models[0]                       # primary (used in names/logs)
        self.strategy = os.getenv("PAATTI_LLM_STRATEGY", "round_robin")
        self.max_retries = int(os.getenv("PAATTI_LLM_RETRIES", "4"))
        self.verbose = True
        self.name = "gemini:" + ",".join(self.models)
        self.last_model = None                            # which model answered the last request
        self.temperature = temperature
        self.thinking_level = os.getenv("PAATTI_THINKING_LEVEL", "low")
        self._schema_ok = True                            # flips off if the API rejects response_schema
        self._no_thinking: set = set()                    # models that reject thinking_config
        rpm = float(os.getenv("PAATTI_LLM_RPM", "0") or 0)  # client-side throttle per model; 0 = off
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0

    # ---------------------------------------------------------------- config
    @staticmethod
    def _gen(model: str) -> int:
        m = re.search(r"gemini-(\d+)", model)
        return int(m.group(1)) if m else 3

    def _config(self, model, system=None, json_schema=None, json_mode=False):
        from google.genai import types
        kw = {}
        if system:
            kw["system_instruction"] = system
        if self._gen(model) >= 3:
            if model not in self._no_thinking:
                kw["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        else:
            kw["temperature"] = self.temperature
            if "2.5" in model:
                kw["thinking_config"] = types.ThinkingConfig(thinking_budget=int(os.getenv("PAATTI_THINKING_BUDGET", "512")))
        if json_schema is not None or json_mode:
            kw["response_mime_type"] = "application/json"
        if json_schema is not None:
            kw["response_schema"] = json_schema
        return types.GenerateContentConfig(**kw)

    # ---------------------------------------------------------------- pool
    def _order(self) -> list:
        """Models to try for this request, available ones first (cooled-down ones are skipped)."""
        import time
        now = time.time()
        with GeminiLLM._lock:
            avail = [m for m in self.models if GeminiLLM._cooldown.get(m, 0) <= now]
            if self.strategy == "round_robin" and avail:
                k = GeminiLLM._rr % len(avail)
                GeminiLLM._rr += 1
                avail = avail[k:] + avail[:k]
        return avail

    def _cool(self, model, until, why):
        with GeminiLLM._lock:
            GeminiLLM._cooldown[model] = max(GeminiLLM._cooldown.get(model, 0), until)
        if self.verbose:
            print(f"    {model}: {why}; switching to the next model", flush=True)

    def pool_status(self) -> dict:
        import time
        now = time.time()
        return {m: ("ok" if GeminiLLM._cooldown.get(m, 0) <= now
                    else f"cooling {int(GeminiLLM._cooldown[m] - now)}s") for m in self.models}

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

    def _send(self, contents, make_config):
        """Try models from the pool until one answers. make_config(model) builds that model's config."""
        import time
        from google.genai import errors
        order = self._order()
        if not order:
            raise DailyQuotaExceeded("Daily free-tier quota used up for every model in the pool "
                                     f"({', '.join(self.models)}). It resets at midnight Pacific time.")
        retries = self.max_retries if len(order) == 1 else 1   # with alternatives, fail over fast
        last, all_daily = None, True
        for model in order:
            delay, attempt = 4.0, 0
            config = make_config(model)
            while True:
                attempt += 1
                self._throttle(model)
                try:
                    r = self.client.models.generate_content(model=model, contents=contents, config=config)
                    self.last_model = model
                    return r
                except errors.APIError as e:
                    code = getattr(e, "code", None)
                    if code == 400 and "thinking" in str(e).lower() and getattr(config, "thinking_config", None):
                        self._no_thinking.add(model)              # resend this model without thinking
                        config = config.model_copy(update={"thinking_config": None})
                        continue
                    if code == 404:
                        last, all_daily = e, False
                        self._cool(model, time.time() + 86400, "not available to this API key (404)")
                        break
                    if code not in self._TRANSIENT:
                        raise
                    last = e
                    if code == 429 and self._is_daily_quota(e):
                        self._cool(model, _next_pacific_midnight(), "daily free-tier quota used up")
                        break
                    all_daily = False
                except Exception as e:  # timeouts / connection resets
                    if "timeout" not in type(e).__name__.lower() and "connect" not in type(e).__name__.lower():
                        raise
                    last, code, all_daily = e, "timeout", False
                if attempt > retries:
                    self._cool(model, time.time() + 60, f"{code} after {attempt} tries")
                    break
                m = re.search(r"retry(?:Delay| in)[^0-9]*([0-9.]+)\s*s", str(last))
                wait = min(float(m.group(1)) + 1, 90) if m else delay
                if self.verbose:
                    print(f"    Gemini {code} on {model}: retry {attempt}/{retries} in {wait:.0f}s", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 60)
        if all_daily:
            raise DailyQuotaExceeded("Daily free-tier quota used up for every model in the pool "
                                     f"({', '.join(order)}). It resets at midnight Pacific time.")
        if isinstance(last, errors.ClientError) and getattr(last, "code", None) == 404 and len(order) == 1:
            raise RuntimeError(f"Gemini model '{order[0]}' is not available to this key ({last}). "
                               "Set PAATTI_LLM in .env to a current model id.") from last
        raise last

    # ---------------------------------------------------------------- calls
    def _call(self, contents, system=None, schema=None):
        """Graceful degradation: response_schema -> JSON mode -> plain text."""
        from google.genai import errors
        modes = []
        if schema is not None and self._schema_ok:
            modes.append(("schema", lambda m: self._config(m, system, json_schema=schema)))
        if schema is not None:
            hint = ("\n\nReturn ONLY a JSON object with keys: " + ", ".join(schema.model_fields)) if hasattr(schema, "model_fields") else ""
            modes.append(("json", lambda m: self._config(m, (system or "") + hint, json_mode=True)))
        modes.append(("plain", lambda m: self._config(m, system)))
        last = None
        for mode, make in modes:
            try:
                return self._send(contents, make)
            except errors.ClientError as e:
                last = e
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
        from google.genai import errors
        try:
            r = self._send(prompt, lambda m: self._config(m, json_mode=True))
        except errors.ClientError as e:
            if getattr(e, "code", None) != 400:
                raise
            r = self._send(prompt, lambda m: self._config(m))
        return _loose_json(r.text)

    def transcribe(self, audio, mime):
        from google.genai import types
        r = self._send([types.Part.from_bytes(data=audio, mime_type=mime or "audio/wav"),
                        "Transcribe this speech exactly as spoken, in whatever language(s) it is in. Write each "
                        "language in its OWN script: Tamil in Tamil script (தமிழ்), Hindi in Devanagari, Malayalam, "
                        "Telugu, Kannada, Bengali, Urdu in theirs; use Latin letters only for English words. Never "
                        "romanize an Indian language. Return only the transcript, nothing else. If there is no "
                        "intelligible speech (silence, noise, music), return an empty string."],
                       lambda m: self._config(m))
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
