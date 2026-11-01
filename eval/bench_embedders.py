"""Benchmark RAW embedding models on the v2 knowledge base (no fine-tuned weights).

  python eval/bench_embedders.py                  # all 'core' models + BM25 baseline
  python eval/bench_embedders.py --extra          # also LaBSE, EmbeddingGemma (gated), jina-v3
  python eval/bench_embedders.py --gemini         # also Gemini API embeddings (needs GOOGLE_API_KEY)
  python eval/bench_embedders.py --models bge-m3 e5-large

For every model it reports dense-only and hybrid (BM25 + dense, RRF) scores on the
143 advice queries, split by English / Tanglish / Tamil script, with a 95% bootstrap
interval on Hit@1 so you can tell real gaps from noise. Output:
  eval/results_embedders.md   (table)   eval/results_embedders.json (full, incl. misses)
"""
from __future__ import annotations
import argparse, gc, json, platform, re, shutil, sys, time, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paatti.embedders import MODELS
from paatti.kb import KnowledgeBase
from paatti.retrieval import HybridRetriever, make_embedder

ROWS = [json.loads(l) for l in (ROOT / "eval" / "eval_set.jsonl").read_text().splitlines() if l.strip()]
ADVICE = [r for r in ROWS if r["type"] == "advice"]
TA = re.compile(r"[஀-௿]")


def lang(r):
    if TA.search(r["query"]):
        return "tamil"
    return "tanglish" if r["lang"] == "ta-mixed" else "english"


def ranks_for(ret, rows):
    out, lat = [], []
    for r in rows:
        t = time.perf_counter()
        ranked = [n for n, _ in ret.search(r["query"], k=109)]
        lat.append(time.perf_counter() - t)
        out.append(next((i + 1 for i, n in enumerate(ranked) if n in r["gold"]), 999))
    return np.array(out), float(np.mean(lat) * 1000)


def summarize(ranks, rows, seed=0):
    def m(rk):
        return {"hit@1": float(np.mean(rk == 1)), "hit@3": float(np.mean(rk <= 3)),
                "hit@5": float(np.mean(rk <= 5)),
                "hit@8": float(np.mean(rk <= 8)), "mrr": float(np.mean(1.0 / rk)), "n": int(len(rk))}
    res = m(ranks)
    rng = np.random.default_rng(seed)
    boots = [np.mean(ranks[rng.integers(0, len(ranks), len(ranks))] == 1) for _ in range(1000)]
    res["hit@1_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    langs = np.array([lang(r) for r in rows])
    res["by_lang"] = {l: m(ranks[langs == l]) for l in ("english", "tanglish", "tamil") if (langs == l).any()}
    return res


def misses(ranks, rows, ret):
    out = []
    for rk, r in zip(ranks, rows):
        if rk > 3:
            out.append({"q": r["query"], "gold": r["gold"], "rank": int(rk),
                        "top3": [n for n, _ in ret.search(r["query"], k=3)]})
    return out


def run_one(kb, spec, results):
    t0 = time.time()
    emb = make_embedder(spec)
    load_s = time.time() - t0
    t1 = time.time()
    dense = HybridRetriever(kb, emb, use_bm25=False)  # encodes + caches doc vectors
    enc_s = time.time() - t1
    hybrid = HybridRetriever(kb, emb, bm25_weight=1.0)   # reuses cached vectors
    hybrid_w = HybridRetriever(kb, emb, bm25_weight=0.3)
    for label, ret in (("dense", dense), ("hybrid", hybrid), ("hybrid-0.3", hybrid_w)):
        rk, ms = ranks_for(ret, ADVICE)
        s = summarize(rk, ADVICE)
        s.update(model=spec, mode=label, query_ms=round(ms, 1), load_s=round(load_s, 1),
                 encode_docs_s=round(enc_s, 1), dim=int(dense.doc_vecs.shape[1]),
                 note=MODELS.get(spec, {}).get("note", ""), misses=misses(rk, ADVICE, ret))
        results.append(s)
        print(f"  {label:10s} hit@8={s['hit@8']:.3f} hit@1={s['hit@1']:.3f} [{s['hit@1_ci'][0]:.2f}-{s['hit@1_ci'][1]:.2f}] "
              f"hit@3={s['hit@3']:.3f} hit@5={s['hit@5']:.3f} mrr={s['mrr']:.3f} "
              f"| en={s['by_lang']['english']['hit@1']:.2f} "
              f"tanglish={s['by_lang'].get('tanglish', {}).get('hit@1', 0):.2f} "
              f"tamil={s['by_lang'].get('tamil', {}).get('hit@1', 0):.2f} | {ms:.0f} ms/q", flush=True)
    del emb, dense, hybrid, hybrid_w
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def free_gb(path=ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def cached_repo_ids() -> set:
    try:
        from huggingface_hub import scan_cache_dir
        return {r.repo_id for r in scan_cache_dir().repos}
    except Exception:
        return set()


def purge_model(model_id: str) -> float:
    """Delete one model from the HF cache. Returns GB freed."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        revs = [rev.commit_hash for repo in info.repos if repo.repo_id == model_id for rev in repo.revisions]
        if not revs:
            return 0.0
        strat = info.delete_revisions(*revs)
        strat.execute()
        return strat.expected_freed_size / 1e9
    except Exception as e:
        print(f"  (could not purge {model_id}: {e})")
        return 0.0


def write_reports(results, failed):
    try:
        _write_reports(results, failed)
    except OSError as e:
        print(f"!! could not save results ({e}). Free some disk space; partial results printed above.")


def _write_reports(results, failed):
    (ROOT / "eval" / "results_embedders.json").write_text(json.dumps(
        {"platform": platform.platform(), "n_queries": len(ADVICE), "results": results, "failed": failed},
        ensure_ascii=False, indent=1))
    rows = sorted(results, key=lambda r: -r["mrr"])
    L = [f"# Raw embedding benchmark ({len(ADVICE)} advice queries, v2 KB)\n",
         "| model | mode | Hit@1 | 95% CI | Hit@3 | Hit@5 | Hit@8* | MRR | EN@1 | Tanglish@1 | Tamil@1 | dim | ms/query | notes |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bl = r["by_lang"]
        L.append(f"| {r['model']} | {r['mode']} | {r['hit@1']:.3f} | {r['hit@1_ci'][0]:.2f}–{r['hit@1_ci'][1]:.2f} | "
                 f"{r['hit@3']:.3f} | {r['hit@5']:.3f} | {r['hit@8']:.3f} | {r['mrr']:.3f} | {bl['english']['hit@1']:.2f} | "
                 f"{bl.get('tanglish', {}).get('hit@1', 0):.2f} | {bl.get('tamil', {}).get('hit@1', 0):.2f} | "
                 f"{r.get('dim', '-')} | {r['query_ms']} | {r.get('note', '')} |")
    if failed:
        L.append("\n**Failed to load:** " + "; ".join(f"{k}: {v}" for k, v in failed.items()))
    L.append("\n*Hit@8 = how often the right verse is among the 8 candidates Gemini sees (what matters for the chatbot).")
    L.append("\nOverlapping 95% CIs = no real difference. Tanglish/Tamil subsets are small (n≈6 each): treat as indicative.")
    (ROOT / "eval" / "results_embedders.md").write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", help="registry keys or st:<hf id>")
    p.add_argument("--extra", action="store_true", help="include optional/gated models")
    p.add_argument("--gemini", action="store_true", help="include Gemini API embeddings")
    p.add_argument("--cached", action="store_true",
                   help="only models already in your HF cache (no downloads, no extra disk)")
    p.add_argument("--purge-cache", action="store_true",
                   help="delete each model from the HF cache after benchmarking it (models you already had are kept)")
    a = p.parse_args()
    cache = cached_repo_ids()
    specs = a.models or [k for k, m in MODELS.items() if m["group"] == "core" or a.extra]
    if a.cached:
        specs = [k for k in (a.models or MODELS) if MODELS.get(k, {}).get("id") in cache]
        print("cached models:", ", ".join(specs) or "none")
    else:  # cached models first: they cost nothing and give results early
        specs = sorted(specs, key=lambda k: MODELS.get(k, {}).get("id") not in cache)
    print(f"free disk: {free_gb():.1f} GB  (largest model needs ~2.5 GB; all core models ~15 GB without --purge-cache)")
    if free_gb() < 3 and not a.cached:
        print("!! Less than 3 GB free: model downloads will fail. Free space first (see README > Benchmark).")
        if not a.models:
            sys.exit(1)
    preexisting = cache
    if a.gemini:
        specs.append("gemini")
    kb = KnowledgeBase()
    results, failed = [], {}

    print("== bm25 (baseline)")
    bm = HybridRetriever(kb, None)
    rk, ms = ranks_for(bm, ADVICE)
    s = summarize(rk, ADVICE); s.update(model="bm25", mode="lexical", query_ms=round(ms, 1), dim="-",
                                        note="no model", misses=misses(rk, ADVICE, bm))
    results.append(s)
    print(f"  hit@1={s['hit@1']:.3f} mrr={s['mrr']:.3f}")

    for spec in specs:
        print(f"== {spec}", flush=True)
        try:
            run_one(kb, spec, results)
        except Exception as e:
            failed[spec] = f"{type(e).__name__}: {str(e)[:160]}"
            print("  FAILED:", failed[spec]); traceback.print_exc(limit=1)
        if a.purge_cache and spec != "gemini":
            mid = MODELS.get(spec, {}).get("id", spec.removeprefix("st:"))
            if mid not in preexisting:
                print(f"  purged {mid}: {purge_model(mid):.1f} GB freed, {free_gb():.1f} GB free")
        write_reports(results, failed)  # after every model, so partial runs are kept
    if not specs:
        write_reports(results, failed)


if __name__ == "__main__":
    main()
