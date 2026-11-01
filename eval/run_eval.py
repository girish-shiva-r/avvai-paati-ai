"""Benchmark harness.

Retrieval benchmark (old docs vs new docs, every embedder):
  python eval/run_eval.py retrieval --embedders none bge-m3 e5-large gemini
  (for the full raw-model comparison use eval/bench_embedders.py)

Full pipeline (needs GOOGLE_API_KEY):
  python eval/run_eval.py pipeline --embedder gemini [--judge] [--limit 60]
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paatti.kb import KnowledgeBase
from paatti.retrieval import HybridRetriever, make_embedder

EVAL = [json.loads(l) for l in (ROOT / "eval" / "eval_set.jsonl").read_text().splitlines() if l.strip()]


def retrieval_metrics(ret, rows):
    h1 = h3 = h5 = mrr = 0
    misses = []
    for r in rows:
        ranked = [n for n, _ in ret.search(r["query"], k=20)]
        rank = next((i + 1 for i, n in enumerate(ranked) if n in r["gold"]), None)
        h1 += rank == 1; h3 += bool(rank and rank <= 3); h5 += bool(rank and rank <= 5)
        mrr += 1 / rank if rank else 0
        if not rank or rank > 5:
            misses.append({"q": r["query"], "gold": r["gold"], "top5": ranked[:5]})
    n = len(rows)
    return {"hit@1": h1 / n, "hit@3": h3 / n, "hit@5": h5 / n, "mrr": mrr / n, "n": n}, misses


def cmd_retrieval(a):
    kb = KnowledgeBase()
    rows = [r for r in EVAL if r["type"] == "advice"]
    results, all_misses = [], {}
    for spec in a.embedders:
        emb = make_embedder(spec)
        variants = [("bm25", None, True)] if emb is None else [("dense", emb, False), ("hybrid", emb, True)]
        for label, e, use_bm25 in variants:
            for mode in a.docs:
                t = time.time()
                ret = HybridRetriever(kb, e, doc_mode=mode, use_bm25=use_bm25)
                m, misses = retrieval_metrics(ret, rows)
                name = f"{spec if spec != 'none' else 'bm25'} | {label} | docs={mode}"
                m["name"], m["secs"] = name, round(time.time() - t, 1)
                results.append(m); all_misses[name] = misses
                print(f"{name:85s} hit@1={m['hit@1']:.3f} hit@3={m['hit@3']:.3f} hit@5={m['hit@5']:.3f} mrr={m['mrr']:.3f}", flush=True)
    # deterministic lookup router
    lk = [r for r in EVAL if r["type"] == "lookup"]
    acc = sum(kb.lookup(r["query"]) in r["gold"] for r in lk) / len(lk)
    false_trig = sum(kb.lookup(r["query"]) is not None for r in rows) / len(rows)
    print(f"\nlookup router: accuracy={acc:.2f} on {len(lk)} lookup queries; false triggers on advice={false_trig:.3f}")
    out = ROOT / "eval" / "results_retrieval.json"
    out.write_text(json.dumps({"results": results, "misses": all_misses, "lookup_acc": acc,
                               "lookup_false_trigger": false_trig}, ensure_ascii=False, indent=1))
    print("saved", out)


JUDGE = """You are grading a chatbot that gives life advice as the Tamil poet Avvaiyar using Aathichoodi.
User message: {q}
Verse chosen: {v}
Bot reply: {r}
Score each 1-5: relevance (verse+advice fit the user's actual problem), faithfulness (explanation true to the
verse's meaning given), warmth (grandmotherly, kind, not preachy), actionability (concrete next steps).
Return JSON: {{"relevance":int,"faithfulness":int,"warmth":int,"actionability":int}}"""


def cmd_pipeline(a):
    from paatti.engine import Paatti
    from paatti.llm import make_llm
    if a.rpm:
        os.environ["PAATTI_LLM_RPM"] = str(a.rpm)  # stay under the free tier's per-minute limit
    bot = Paatti(llm=make_llm(a.llm), embedder_spec=a.embedder)
    judge_llm = None
    if a.judge:
        judge_llm = bot.llm if a.judge_llm == "same" else make_llm(a.judge_llm)
        if judge_llm is not bot.llm:
            judge_llm.max_retries = 1  # the judge must never stall the run; missed scores can be filled later
        print(f"judge: {judge_llm.name} (separate model = separate free-tier quota, and no self-grading)")
    judge_fails = 0
    # preflight: fail fast instead of scoring 159 fallback answers
    try:
        bot.llm.reply("Reply with JSON.", [{"role": "user", "content": "hi paatti"}])
        print(f"LLM ok: {bot.llm.name}")
    except Exception as e:
        sys.exit(f"LLM preflight failed for {bot.llm.name}: {e}")
    rows = EVAL if not a.limit else EVAL[: a.limit] + [r for r in EVAL[a.limit:] if r["type"] != "advice"]

    # results are appended line by line, so Ctrl+C / crashes lose nothing; --resume continues a run
    out = Path(a.resume) if a.resume else ROOT / "eval" / f"results_pipeline_{int(time.time())}.jsonl"
    done = {}
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                o = json.loads(l)
                if o.get("intent") not in ("fallback", "error") and o.get("llm", bot.llm.name) == bot.llm.name:
                    done[o["id"]] = o  # failed rows, and rows from a different model, get rerun
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to run -> {out}")
    t0 = time.time()
    with out.open("a") as f:
        for i, r in enumerate(todo, 1):
            try:
                o = bot.answer(r["query"])
            except Exception as e:
                o = {"intent": "error", "error": str(e)[:300], "verse_no": None, "response": ""}
            if "Daily free-tier quota" in str(o.get("error", "")):
                print(f"\n!! {o['error']}\nStopped. Progress is saved; continue later with:\n"
                      f"   python3 eval/run_eval.py pipeline --embedder {a.embedder} --llm {a.llm}"
                      f"{' --judge' if a.judge else ''} --resume {out}")
                break
            o.update(id=r["id"], llm=bot.llm.name, query=r["query"], gold=r["gold"], type=r["type"],
                     correct=_correct(r, o), expect_script=r.get("expect_script"))
            if judge_llm and r["type"] == "advice" and o.get("verse") and o.get("intent") not in ("fallback", "error"):
                o["judge"] = judge(judge_llm, r["query"], o)
                judge_fails = judge_fails + 1 if o["judge"] is None else 0
                if getattr(judge_llm, "_quota_out", False) or judge_fails >= 3:
                    print(f"   (judge unavailable: continuing without scores; fill them in later with:\n"
                          f"    python3 eval/run_eval.py rejudge {out})")
                    judge_llm = None
            f.write(json.dumps(o, ensure_ascii=False) + "\n"); f.flush()
            done[r["id"]] = o
            tag = "ERR" if o["intent"] in ("fallback", "error") else ("OK " if o["correct"] else "XX ")
            print(f"[{tag}] {i}/{len(todo)} {(time.time() - t0) / 60:4.1f}m {r['type']:8s} v={o.get('verse_no')} "
                  f"gold={r['gold']} {r['query'][:50]}", flush=True)
            if a.sleep:
                time.sleep(a.sleep)
    summarize_pipeline([done[r["id"]] for r in rows if r["id"] in done], a.judge, out)


def _correct(r, o):
    t = r["type"]
    if t in ("advice", "lookup"):
        return o.get("verse_no") in r["gold"]
    if t == "offtopic":
        return o.get("intent") in ("offtopic", "smalltalk")
    return o.get("intent") == "crisis"


_SCRIPTS = {"devanagari": (0x0900, 0x097F), "malayalam": (0x0D00, 0x0D7F), "telugu": (0x0C00, 0x0C7F),
            "tamil": (0x0B80, 0x0BFF), "kannada": (0x0C80, 0x0CFF), "bengali": (0x0980, 0x09FF)}


def script_ok(text, expect):
    """True if most letters in the reply are in the expected script ('latin' = ASCII/Latin letters)."""
    letters = [c for c in text or "" if c.isalpha()]
    if not letters:
        return False
    if expect == "latin":
        hit = sum(c.isascii() or "\u00c0" <= c <= "\u024f" for c in letters)
    else:
        lo, hi = _SCRIPTS[expect]
        hit = sum(lo <= ord(c) <= hi for c in letters)
    return hit / len(letters) >= 0.6


def summarize_pipeline(outs, use_judge, out):
    print()
    bad = [o for o in outs if o.get("intent") in ("fallback", "error")]
    ok = [o for o in outs if o.get("intent") not in ("fallback", "error")]
    if bad:
        print(f"!! {len(bad)} queries hit API errors and are EXCLUDED below. Fill them in with: --resume {out}")
    for t in ("advice", "lookup", "offtopic", "crisis"):
        xs = [o for o in ok if o["type"] == t]
        if xs:
            c = sum(o["correct"] for o in xs)
            print(f"{t:9s} accuracy {c}/{len(xs)} = {c / len(xs):.3f}")
    ml = [o for o in ok if o.get("expect_script")]
    if ml:
        c = sum(script_ok(o.get("response", ""), o["expect_script"]) for o in ml)
        print(f"language  replied in the requested language/script {c}/{len(ml)} = {c / len(ml):.3f}")
    lat = [o.get("latency", 0) for o in ok if o.get("latency")]
    if lat:
        print(f"latency p50={statistics.median(lat):.2f}s p95={sorted(lat)[int(.95 * (len(lat) - 1))]:.2f}s")
    if use_judge:
        keys = ("relevance", "faithfulness", "warmth", "actionability")
        js = [o["judge"] for o in ok if isinstance(o.get("judge"), dict) and all(k in o["judge"] for k in keys)]
        for k in keys:
            if js:
                print(f"judge {k:13s} {statistics.mean(float(j[k]) for j in js):.2f}/5  (n={len(js)})")
    print("saved", out)


def judge(llm, q, o):
    v = o["verse"]
    try:
        return llm.generate_json(JUDGE.format(q=q, v=f"{v['no']}. {v['translit']}: {v['meaning']}", r=o["response"]))
    except Exception as e:
        if "Daily free-tier quota" in str(e):
            llm._quota_out = True
        print("  judge failed:", str(e)[:200])
        return None


def cmd_rejudge(a):
    """Add judge scores to rows of a results file that don't have them (e.g. the judge was overloaded)."""
    from paatti.llm import make_llm
    if a.rpm:
        os.environ["PAATTI_LLM_RPM"] = str(a.rpm)
    llm = make_llm(a.judge_llm)
    path = Path(a.file)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    todo = [o for o in rows if o.get("type") == "advice" and o.get("verse") and not isinstance(o.get("judge"), dict)
            and o.get("intent") not in ("fallback", "error")]
    print(f"{len(todo)} rows need scores; judge = {llm.name}")
    fails = 0
    try:
        for i, o in enumerate(todo, 1):
            o["judge"] = judge(llm, o["query"], o)
            fails = fails + 1 if o["judge"] is None else 0
            print(f"  {i}/{len(todo)} {'ok' if o['judge'] else 'failed'}  {o['query'][:50]}", flush=True)
            if fails >= 3 or getattr(llm, "_quota_out", False):
                print("judge keeps failing; stopping (scores so far are saved). Try later or another --judge-llm.")
                break
    finally:  # always write back what we have
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in rows))
        tmp.replace(path)
    summarize_pipeline(rows, True, path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("retrieval")
    r.add_argument("--embedders", nargs="+", default=["none"])
    r.add_argument("--docs", nargs="+", default=["legacy", "v2"])
    q = sp.add_parser("pipeline")
    q.add_argument("--embedder", default="gemini")
    q.add_argument("--llm", default="gemini", help="gemini (default model) or gemini:<model id>")
    q.add_argument("--judge-llm", default="gemini:gemini-3.1-flash-lite",
                   help="model that scores answers; 'same' = reuse --llm")
    q.add_argument("--rpm", type=float, default=9, help="max requests/minute per model (free tier: 10 for Flash-Lite)")
    q.add_argument("--judge", action="store_true")
    q.add_argument("--limit", type=int, default=0)
    q.add_argument("--resume", help="path of an earlier results_pipeline_*.jsonl to continue")
    q.add_argument("--sleep", type=float, default=0, help="seconds to wait between queries (free-tier rate limits)")
    j = sp.add_parser("rejudge", help="fill in missing judge scores in a results file")
    j.add_argument("file")
    j.add_argument("--judge-llm", default="gemini:gemini-3.1-flash-lite")
    j.add_argument("--rpm", type=float, default=9)
    a = p.parse_args()
    {"retrieval": cmd_retrieval, "pipeline": cmd_pipeline, "rejudge": cmd_rejudge}[a.cmd](a)
