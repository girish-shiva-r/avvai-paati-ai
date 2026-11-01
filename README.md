# Avvaiyar Paatti AI — v2

A chatbot that gives life advice through the 109 verses of Avvaiyar's *Aathichoodi* (ஆத்திசூடி).
English, தமிழ் and Tanglish in; a verse card plus warm, practical advice out.

## What changed from v1 and why

| Problem in v1 | Fix in v2 |
|---|---|
| Retrieval docs were long, jargon-heavy paragraphs ("transformational integrity… circular economy") that don't resemble how users talk | **Curated knowledge base** (`data/aathichoodi_kb.json`): every verse has a corrected literal gloss, a plain meaning, themes, and 5–6 everyday *situations* written in the user's voice (incl. Tanglish) |
| One vector per verse | **Multi-vector index**: meaning + each situation is its own document; a verse scores by its best match |
| Dense-only or keyword-hack retrieval (`intent_boost` map) | **Hybrid retrieval**: BM25 + embeddings fused with reciprocal rank fusion. No hand-written keyword map |
| LLM got only top-1 (Space) or no verse number (local) and had to "quote the verse", so it hallucinated | Gemini sees the top-8 candidates **and the full list of 109**, returns JSON `{intent, verse_no, response}`; **verse text/number always come from the KB** and are validated |
| E5 models used without `query:`/`passage:` prefixes; Space didn't load the fine-tuned model | Prefixes handled automatically for any `st:` model |
| Some source translations wrong (#13, #35, #43, #96, #100 …) | Corrected `literal` + `meaning` fields (original kept as `source_english`) |
| `gemini-2.0-flash` + `google-generativeai` are retired | `google-genai` SDK, `gemini-3.8-flash` (configurable; 2.5-flash is also closed to new users now) |
| Single-turn only | Conversation history; follow-ups ("what should I do then?", "another verse") work |
| No direct verse questions | Deterministic lookup: "verse 54", "explain ஆறுவது சினம்", "nandri maravel meaning", "last verse" |
| No safety handling | Crisis messages short-circuit to helplines (Tele-MANAS 14416) before any verse |
| No measurement | **Labeled eval set** (159 queries) + benchmark harness |

## Results so far

Measured on 143 advice queries that were written separately from the KB situations:

| Retriever | Docs | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| BM25 | v1 (`Embedding_Text`) | 0.070 | 0.175 | 0.224 | 0.151 |
| BM25 | **v2 KB** | **0.517** | **0.671** | **0.734** | **0.624** |

Lookup router: 10/10 correct, 0 false triggers on advice queries.

**Raw embedding models on the v2 KB** (MacBook Air run, 31 Oct 2026; no fine-tuning):

| Model | Mode | Hit@1 | 95% CI | Hit@3 | Hit@5 | MRR | Tanglish@1 (n=8) | Tamil@1 (n=4) |
|---|---|---|---|---|---|---|---|---|
| paraphrase-multilingual-mpnet | dense | **0.755** | 0.69–0.83 | 0.902 | **0.937** | 0.834 | 0.50 | 0.75 |
| **bge-m3** (default) | dense | **0.748** | 0.68–0.82 | 0.902 | 0.930 | 0.834 | 0.75 | 0.50 |
| e5-large-v2 (English-only) | dense | 0.699 | 0.62–0.77 | 0.881 | 0.916 | 0.795 | 0.88 | 0.00 |
| Vyakyarth | dense | 0.664 | 0.58–0.74 | 0.832 | 0.923 | 0.769 | 0.62 | 0.50 |
| bge-m3 | hybrid (BM25 equal weight) | 0.650 | 0.57–0.73 | 0.839 | 0.874 | 0.751 | 0.75 | 0.00 |
| multilingual-e5-large | hybrid | 0.650 | 0.57–0.73 | 0.832 | 0.881 | 0.756 | 0.75 | 0.00 |
| l3cube Tamil SBERT | dense | 0.580 | 0.50–0.66 | 0.804 | 0.853 | 0.709 | 0.75 | 0.25 |
| BM25 only | keyword | 0.510 | 0.43–0.59 | 0.685 | 0.727 | 0.620 | 0.62 | 0.00 |
| MuRIL sentence-transformer | dense | 0.126 | 0.07–0.19 | 0.196 | 0.224 | 0.197 | 0.12 | 0.50 |

Takeaways: (1) mpnet and bge-m3 are statistically tied; bge-m3 is the default for its stronger multilingual
coverage, mpnet-ml is the lighter alternative (half the size). (2) Equal-weight BM25 fusion *hurts* strong dense
models, so retrieval is now dense-only by default (`PAATTI_BM25_WEIGHT`), and BM25 is skipped whenever it has
no keyword overlap (Tamil script). (3) The right verse is in the top 5 ~93% of the time and Gemini sees the
top 8 plus the full list, so retrieval is no longer the bottleneck. Fine-tuning is not needed yet.
Not yet run: gte-multilingual, e5-large-instruct, arctic-embed-l-v2, Qwen3-Embedding (disk space).

**End-to-end (Gemini 3.5 Flash-Lite + bge-m3 retrieval, full 159-question eval, 1 Nov 2026):**

| | Result |
|---|---|
| Advice: right verse chosen (143) | **0.937** (old system's retrieval: 0.07 at rank 1) |
| Verse lookups (10) / off-topic (4) / crisis (2) | 10/10 · 4/4 · 2/2 |
| Judge (gemini-3.1-flash-lite, 1–5) | relevance 4.78 · faithfulness 4.86 · warmth 5.0 · actionability 4.44 (n=85) |
| Latency | p50 5.1 s, p95 7.4 s |

Known weak spot: Tamil-script queries (n=4) are the least reliable group.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add GOOGLE_API_KEY
streamlit run app.py            # runs the engine in-process, no backend needed
# or: uvicorn api:app --port 7860   then set API_URL=http://localhost:7860 for Streamlit
python -m pytest -q             # offline tests (no keys needed)
```

## Benchmark

**Step 1: raw embedding models only.** The old fine-tuned checkpoints (`GSR-608001/*`) were trained on the v1
documents, so they are excluded. Fine-tuning comes later, and only if the raw models fall short.

```bash
pip install -r requirements-eval.txt
python eval/bench_embedders.py                 # 10 core raw models + BM25 baseline (auto uses mps/cuda)
python eval/bench_embedders.py --extra --gemini  # + LaBSE, EmbeddingGemma (gated), jina-v3, Gemini API
python eval/bench_embedders.py --models bge-m3 e5-base   # just a few
```
Models and their exact query/document prompts live in `paatti/embedders.py`.
Output: `eval/results_embedders.md` (table: dense and hybrid, Hit@1 with 95% CI, English/Tanglish/Tamil split,
ms/query) and `eval/results_embedders.json` (incl. each model's misses).
No GPU / Mac trouble? Zip this folder as `v2.zip` and run `eval/bench_colab.ipynb` on Colab or Kaggle.

**Low on disk?** All core models together are ~15 GB in `~/.cache/huggingface`. Use
`python eval/bench_embedders.py --purge-cache`: each model is deleted right after it is scored, so you only need
~3 GB free at a time (models you had cached before the run are kept). See what's using space with
`df -h ~` and `du -sh ~/.cache/huggingface/hub/* | sort -h`; old checkpoints can be removed with `hf cache delete`.

**Step 2: end-to-end with Gemini** (+ LLM-judge scores for relevance/faithfulness/warmth/actionability):
```bash
python eval/run_eval.py pipeline --embedder bge-m3 --judge     # use the step-1 winner
```
Free-tier friendly: answers use `gemini-3.5-flash-lite`, the judge uses `gemini-3.1-flash-lite` (separate quota,
no self-grading), and requests are paced at 9/min. A full run is ~320 requests (~20 min). Results are saved after
every question; if a daily quota runs out the script stops and prints the `--resume` command.
Don't add eval queries to the KB situations; that leaks the test set.

**When to fine-tune:** only if the best hybrid Hit@3 is clearly below what you need *and* the misses are
retrieval failures (gold not in top-8) rather than ambiguous labels. Then train on KB situations, never on `eval_set.jsonl`.

## Deploy (live): Streamlit Community Cloud only

The Streamlit app runs the whole bot in-process; no separate backend or Hugging Face Space is needed
(Hugging Face is only the public download source for the embedding model).

1. Push this folder to GitHub (`girish-shiva-r/avvai-paati-ai`). `.env` is git-ignored, so your key stays local.
2. On [share.streamlit.io](https://share.streamlit.io) → **Create app** → pick the repo, branch `main`, main file `app.py`,
   Python 3.11 or 3.12.
3. **Advanced settings → Secrets**:
   ```toml
   GOOGLE_API_KEY = "your-gemini-key"
   PAATTI_EMBEDDER = "mpnet-ml"
   PAATTI_LLM = "gemini-3.5-flash-lite"
   ```
4. Deploy. The first start downloads mpnet (~1 GB) and embeds the 670 KB texts (about a minute); later visits are instant.

Why mpnet-ml and not bge-m3: they are statistically tied (Hit@1 0.755 vs 0.748; right verse in Gemini's top 8:
97% vs 96%), but bge-m3 + PyTorch needs ~3 GB RAM and Streamlit Community Cloud allows ~2.7 GB at most.

Free-tier notes: Gemini Flash-Lite allows ~500 requests/day (one per chat message), enough for a demo. Apps on
Community Cloud sleep after ~12 h without visitors and take a moment to wake. For real traffic, enable billing and
set `PAATTI_LLM = "gemini-3.8-flash"`.

*Optional:* the FastAPI backend (`api.py`, `Dockerfile`, `scripts/deploy_space.sh`) still works on a Hugging Face
Docker Space if you later want a public API or bge-m3; then set `API_URL` in the Streamlit secrets.

## Config

| env | default | notes |
|---|---|---|
| `PAATTI_LLM` | `gemini-3.5-flash-lite` | free tier allows ~500 requests/day for Flash-Lite but only ~20/day for Flash models; switch to `gemini-3.8-flash` once billing is enabled |
| `PAATTI_LLM_FALLBACK` | – | second model tried when the main one is overloaded or out of daily quota |
| `PAATTI_LLM_RPM` | `0` | client-side requests/minute cap |
| `PAATTI_EMBEDDER` | `mpnet-ml` | `none` = BM25 only; any key from `paatti/embedders.py` (e.g. `bge-m3`, `e5-base`) or `st:<hf id>` |
| `PAATTI_DEVICE` | auto | `cpu` / `mps` / `cuda` |
| `PAATTI_BM25_WEIGHT` | `0` | keyword weight in fusion (0 = dense only) |
| `PAATTI_THINKING_LEVEL` | `low` | `low` / `medium` / `high` (Gemini 3+; `minimal` is not supported) |

## Layout
```
paatti/      engine.py (pipeline) · kb.py (KB + lookup) · retrieval.py (BM25/dense/RRF) · embedders.py (raw model registry) · llm.py · safety.py
data/        aathichoodi_kb.json (built by scripts/build_kb.py from source parquet + kb_part*.py)
eval/        eval_set.jsonl · make_eval.py · bench_embedders.py · bench_colab.ipynb · run_eval.py
api.py · app.py · Dockerfile · scripts/deploy_space.sh · tests/
```

## Next steps (in order of payoff)
1. Run the benchmark above and pick the embedder by numbers (Gemini embeddings avoid hosting a 2 GB model).
2. Have a Tamil speaker review `kb_part*.py` meanings/situations; that's the asset that drives quality.
3. Collect real user queries (log `query`, `verse_no`, thumbs up/down) and grow the eval set from them.
4. Only then, re-fine-tune an embedder on KB situations (MultipleNegativesRanking, hard negatives from confusable verses like 2/32/105 or 12/43/109) and keep the eval set held out.
