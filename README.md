# Avvaiyar Paatti AI

A chatbot grandmother who gives life advice through the 109 verses of Avvaiyar's *Aathichoodi* (ஆத்திசூடி).
Type or speak in any language: English, தமிழ், Tanglish, हिन्दी, മലയാളം, తెలుగు and more. You get back a verse card
with the Tamil text, transliteration and meaning, followed by warm, practical advice in your language.

**Live:** Streamlit Community Cloud, deployed from this repo's `main` branch.

## Features

- **Verse-grounded advice.** Gemini chooses the verse, but the verse number, text and meaning always come from the
  curated knowledge base, so it can't invent verses.
- **Any language.** The reply follows the language and script of your current message. "Answer in Hindi" gets a
  reply in Devanagari, and Hinglish gets Hinglish back. Asking "from now on, reply in Tamil" keeps that language
  for the rest of the chat.
- **Voice input.** The 🎤 button sits inside the chat bar. Gemini transcribes speech in the speaker's own script,
  and silence or noise is ignored. Replies are text only.
- **Direct lookups.** For example: "verse 54", "verse fifty four", "explain ஆறுவது சினம்", "nandri maravel meaning",
  "last verse".
- **Conversation memory.** Follow-ups like "what should I do then?" and "give me another verse" work.
- **Safety.** Crisis messages go straight to helplines (Tele-MANAS 14416, 112) without a verse. The check runs as a
  regex first, then the LLM, so messages that are indirect or garbled by voice are also caught.
- **Respectful persona.** Paatti speaks warmly but doesn't use pet names ("chellame", "beta", "dear child"), in any
  language. A filter removes them if the model slips.
- **Model pool.** Requests rotate across several free Gemini models. When one hits its daily limit or is
  overloaded, the bot moves on to the next (see below).

## How it works

```
message ─▶ crisis check (regex) ─▶ verse lookup? (deterministic) ─▶ retrieval: top 8 verses (multilingual embeddings)
        ─▶ Gemini (model pool): JSON {intent, verse_no, reason, response}, sees the top 8 + all 109 verse names
        ─▶ verse validated against the KB ─▶ pet-name / mixed-script clean-up ─▶ verse card + reply
```

| Piece | Choice |
|---|---|
| Knowledge base | `data/aathichoodi_kb.json`: 109 verses with corrected literal gloss, plain meaning, themes, 564 everyday *situations* written in the user's voice (incl. Tanglish) |
| Retrieval | multi-vector (meaning + each situation; a verse scores by its best match), dense only, `paraphrase-multilingual-mpnet-base-v2` (fits Streamlit's ~2.7 GB RAM) |
| LLM | `google-genai` SDK, Gemini 3.x Flash-Lite / Flash pool, structured JSON output |
| Speech-to-text | Gemini audio input (the same key; no extra service) |
| UI | Streamlit ≥ 1.66 (`st.chat_input(accept_audio=True)`), light theme |

## Results (1 Nov 2026)

**End-to-end** (Gemini 3.5 Flash-Lite + mpnet retrieval, labeled eval set of 167 queries):

| | Result |
|---|---|
| Advice: right verse chosen | **138/151 = 91.4%** (the v1 system's retrieval got 7% at rank 1) |
| Verse lookups / off-topic / crisis | 10/10 · 4/4 · 2/2 |
| Reply in the requested language and script (8 multilingual cases) | 7/8 (was 5/8); a fix for the remaining Hinglish miss is in, not yet re-measured |
| LLM judge (gemini-3.1-flash-lite, 1–5, n=151) | relevance 4.85 · faithfulness 4.93 · warmth 4.36 · actionability 4.55 |
| Latency | p50 3.5 s, p95 6.9 s |

Most of the 13 advice "misses" are defensible alternate verses: for "I get angry", #2 (control anger) and #105
(don't speak in anger) both fit. These cases are label ambiguity rather than wrong answers. Warmth dropped from 5.0
when pet names were removed, which was intended.

**Retrieval only:** raw embedding models with no fine-tuning, on 143 advice queries written separately from the KB.

| Model | Hit@1 | 95% CI | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| **paraphrase-multilingual-mpnet** (default) | **0.755** | 0.69–0.83 | 0.902 | 0.937 | 0.834 |
| bge-m3 | 0.748 | 0.68–0.82 | 0.902 | 0.930 | 0.834 |
| e5-large-v2 (English only) | 0.699 | 0.62–0.77 | 0.881 | 0.916 | 0.795 |
| Vyakyarth | 0.664 | 0.58–0.74 | 0.832 | 0.923 | 0.769 |
| l3cube Tamil SBERT | 0.580 | 0.50–0.66 | 0.804 | 0.853 | 0.709 |
| BM25 (keywords) | 0.510 | 0.43–0.59 | 0.685 | 0.727 | 0.620 |
| MuRIL sentence-transformer | 0.126 | 0.07–0.19 | 0.196 | 0.224 | 0.197 |
| *v1 docs + BM25* | *0.070* | | *0.175* | *0.224* | *0.151* |

The right verse is in Gemini's top 8 for 97% of queries, so retrieval is no longer the bottleneck and fine-tuning
isn't needed. Adding equal-weight BM25 to a strong dense model hurt (0.650), so BM25 is off by default.

**Voice suite** (`scripts/voice_check.py --suite`) covers English (normal, fast, quiet speech), Hindi, Tamil,
Malayalam, Telugu, spoken verse numbers, off-topic, crisis, silence and noise. Transcription takes ~2 s.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add GOOGLE_API_KEY
streamlit run app.py            # the whole bot runs in-process
python -m pytest -q             # 97 offline tests, no key needed
```

## Deploy: Streamlit Community Cloud

No backend or Hugging Face Space is needed. Hugging Face is only the download source for the embedding model.

1. On [share.streamlit.io](https://share.streamlit.io), click **Create app** and choose repo `girish-shiva-r/avvai-paati-ai`,
   branch `main`, file `app.py`.
2. Under **Advanced settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your-gemini-key"
   PAATTI_EMBEDDER = "mpnet-ml"
   ```
   Don't set `PAATTI_LLM` unless you want to pin one model; setting it switches the pool off.
3. Deploy. The first start downloads mpnet (~1 GB) and embeds the KB, which takes about a minute. Later visits are
   instant. Apps sleep after ~12 h without visitors.

## Model pool (free-tier friendly)

Free-tier Gemini quotas are **per model** (Flash-Lite ~500 requests/day, Flash ~20/day) and reset at midnight
Pacific time. The bot therefore spreads requests over a pool:

```
gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash
```

- **round_robin** (default): each message goes to the next model in the pool, so all the quotas are used together.
- **failover**: always uses the first model and only moves down the list when it fails.
- If a model hits its daily quota, it is skipped until midnight Pacific. If it is overloaded (503/504), it gets one
  retry, then the request moves to the next model and that model rests for 60 s. A model that's unavailable (404)
  is skipped for a day.
- The friendly "Paatti is resting" message appears only after **every** model is out of quota.
- Enable "Show retrieval details" in the sidebar to see which model answered.

## Config

| env | default | notes |
|---|---|---|
| `PAATTI_LLM_MODELS` | the pool above | comma-separated list, in priority order |
| `PAATTI_LLM_STRATEGY` | `round_robin` | or `failover` |
| `PAATTI_LLM` | – | older single-model setting; a comma list also works. Used only if `PAATTI_LLM_MODELS` is unset |
| `PAATTI_LLM_FALLBACK` | – | extra model appended to the pool |
| `PAATTI_LLM_RETRIES` | `4` | retries for a transient error when only one model is left |
| `PAATTI_LLM_RPM` | `0` | client-side requests/minute cap per model (0 = off) |
| `PAATTI_THINKING_LEVEL` | `low` | `low` / `medium` / `high` (Gemini 3+) |
| `PAATTI_EMBEDDER` | `mpnet-ml` | `none` = BM25 only; any key in `paatti/embedders.py` (e.g. `bge-m3`) or `st:<hf id>` |
| `PAATTI_BM25_WEIGHT` | `0` | keyword weight in fusion (0 = dense only) |
| `PAATTI_DEVICE` | auto | `cpu` / `mps` / `cuda` |
| `API_URL` | – | send chat to the optional FastAPI backend instead of running in-process |

## Evaluate

```bash
pip install -r requirements-eval.txt
python eval/bench_embedders.py                       # retrieval benchmark of raw models (--purge-cache if low on disk)
python eval/run_eval.py pipeline --judge             # end-to-end: verse accuracy, language, LLM judge
python eval/run_eval.py pipeline --judge --resume    # continue after a quota stop
python scripts/voice_check.py --suite                # voice: macOS `say` clips -> transcript -> reply
```

The pipeline eval pins one model (`gemini-3.5-flash-lite`) so runs stay comparable. The judge runs on
`gemini-3.1-flash-lite`, so no model grades its own answers. Requests are paced at 9/min, and results are saved
after every question. A full run takes ~340 requests (~25 min).
Never copy eval queries into the KB situations, because that leaks the test set.

## Layout
```
paatti/   engine.py (pipeline, language + persona rules) · kb.py (KB, lookup) · retrieval.py · embedders.py
          llm.py (Gemini model pool, transcription) · safety.py (crisis) · voice.py (transcript clean-up)
data/     aathichoodi_kb.json (built by scripts/build_kb.py from kb_part1-4.py)
eval/     eval_set.jsonl (167) · make_eval.py · run_eval.py · bench_embedders.py · bench_colab.ipynb
app.py (Streamlit) · api.py (optional FastAPI) · scripts/voice_check.py · tests/
```

## Next steps
1. Have a Tamil scholar review the meanings and situations in `scripts/kb_part*.py`. This makes the meanings more
   faithful and settles which verse is "right" in ambiguous cases.
2. Collect real user queries with 👍/👎 feedback and grow the eval set from them.
3. Fine-tune an embedder on KB situations only if real traffic shows retrieval misses.
