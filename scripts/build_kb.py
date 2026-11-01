"""Builds data/aathichoodi_kb.json from the source dataset + curated enrichment."""
import json, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kb_part1 import PART1
from kb_part2 import PART2
from kb_part3 import PART3
from kb_part4 import PART4

ROOT = HERE.parent
src = pd.read_parquet(ROOT / "data" / "source_aathichoodi.parquet")
enrich = {**PART1, **PART2, **PART3, **PART4}
assert sorted(enrich) == list(range(1, 110)), "missing verses"

kb = []
for _, r in src.iterrows():
    n = int(r["எண்"])
    e = enrich[n]
    kb.append({
        "no": n,
        "letter": r["தமிழ் எழுத்து"],
        "tamil": r["தமிழ் வாக்கியம்"].strip(),
        "translit": r["Transliteration"].strip(),
        "tamil_meaning": " ".join(r["Tamil Meaning"].split()),
        "source_english": r["English Translation"].strip(),
        "literal": e["literal"],
        "meaning": e["meaning"],
        "themes": e["themes"],
        "situations": e["situations"],
    })
out = ROOT / "data" / "aathichoodi_kb.json"
out.write_text(json.dumps(kb, ensure_ascii=False, indent=1))
print("wrote", out, len(kb), "verses,", sum(len(v["situations"]) for v in kb), "situations")
