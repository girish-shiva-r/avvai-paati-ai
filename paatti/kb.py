"""Knowledge base loading + deterministic verse lookup."""
from __future__ import annotations
import json, re, difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Verse:
    no: int
    letter: str
    tamil: str
    translit: str
    tamil_meaning: str
    source_english: str
    literal: str
    meaning: str
    themes: list = field(default_factory=list)
    situations: list = field(default_factory=list)

    def card(self) -> dict:
        return {"no": self.no, "tamil": self.tamil, "translit": self.translit,
                "literal": self.literal, "meaning": self.meaning,
                "tamil_meaning": self.tamil_meaning}

    def prompt_block(self) -> str:
        return (f"[{self.no}] {self.tamil} ({self.translit})\n"
                f"  literal: {self.literal}\n  meaning: {self.meaning}\n"
                f"  tamil meaning: {self.tamil_meaning}")

    def compact(self) -> str:
        return f"{self.no}. {self.translit} — {self.literal}"


def _norm_latin(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def _norm_tamil(s: str) -> str:
    return re.sub(r"[^஀-௿]", "", s)


class KnowledgeBase:
    def __init__(self, path: Optional[Path] = None):
        path = Path(path or DATA_DIR / "aathichoodi_kb.json")
        self.verses = [Verse(**v) for v in json.loads(path.read_text())]
        self.by_no = {v.no: v for v in self.verses}
        self._tamil = {v.no: _norm_tamil(v.tamil) for v in self.verses}
        self._latin = {v.no: _norm_latin(v.translit) for v in self.verses}

    def __len__(self):
        return len(self.verses)

    def get(self, no: int) -> Optional[Verse]:
        return self.by_no.get(int(no)) if no is not None else None

    # ---------- deterministic lookup ----------
    _NUM = re.compile(
        r"(?:verse|vers|no\.?|number|num|#|aathichoodi|athichudi|aathisudi|பாடல்|வரி)\s*(?:no\.?|number)?\s*[:#-]?\s*(\d{1,3})\b"
        r"|\b(\d{1,3})\s*(?:st|nd|rd|th)?\s+(?:verse|aathichoodi|athichudi|line)", re.I)

    def lookup(self, query: str) -> Optional[int]:
        """Return a verse number if the user is clearly asking about a specific verse."""
        q = query.strip()
        m = self._NUM.search(q)
        if m:
            n = int(m.group(1) or m.group(2))
            if n in self.by_no:
                return n
        ql = q.lower()
        if re.search(r"\b(last|final)\s+(verse|line|aathichoodi)", ql):
            return 109
        if re.search(r"\bfirst\s+(verse|line|aathichoodi)", ql):
            return 1
        # Tamil script: verse text contained in the query (spacing-insensitive)
        qt = _norm_tamil(q)
        if len(qt) >= 4:
            for n, t in self._tamil.items():
                if t and t in qt:
                    return n
            best = max(self._tamil.items(), key=lambda kv: difflib.SequenceMatcher(None, kv[1], qt).ratio())
            if len(qt) <= len(best[1]) * 1.6 and difflib.SequenceMatcher(None, best[1], qt).ratio() > 0.82:
                return best[0]
        # Transliteration: compare against same-length word windows
        words = _norm_latin(q).split()
        best_n, best_r = None, 0.0
        for n, t in self._latin.items():
            k = len(t.split())
            if k < 2 or len(words) < k:
                continue
            for i in range(len(words) - k + 1):
                r = difflib.SequenceMatcher(None, " ".join(words[i:i + k]), t).ratio()
                if r > best_r:
                    best_n, best_r = n, r
        if best_r >= 0.86:
            return best_n
        return None
