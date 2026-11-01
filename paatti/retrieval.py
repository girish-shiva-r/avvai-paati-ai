"""Hybrid retrieval: BM25 + dense embeddings over multi-vector verse docs, fused with RRF.

Each verse is indexed as several small documents (its meaning/themes + every
'situation'), and a verse's score is the max over its documents. This matches
short, problem-style user queries far better than one long abstract paragraph.
"""
from __future__ import annotations
import hashlib, math, os, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .kb import KnowledgeBase

_STOP = set("""a an the and or but if is are was were be been am i me my mine we our you your he she it they them
his her its their this that these those to of in on at for with from by as so do does did not no can could should
would will just very really too also about what how why when who whom which there here have has had im ive dont
cant feel get got keep always never even still more much any some all one into out up over again then than
""".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+|[஀-௿]+", text.lower())
    out = []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        if t.isascii():  # crude English stemming
            for suf in ("ing", "edly", "ed", "ly", "ies", "es", "s"):
                if len(t) > len(suf) + 3 and t.endswith(suf):
                    t = t[: -len(suf)] + ("y" if suf == "ies" else "")
                    break
        out.append(t)
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            dl = len(d)
            for t in q:
                if t in tf:
                    f = tf[t]
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


# ---------------- embedders ----------------
class Embedder:
    name = "base"
    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        raise NotImplementedError


def _best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class STEmbedder(Embedder):
    """sentence-transformers model with the query/doc prompts from paatti.embedders.MODELS."""
    def __init__(self, spec: str, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        from .embedders import resolve
        m = resolve(spec)
        self.name, self.key, self.model_id = f"st:{m['id']}", m["key"], m["id"]
        self.q_prompt, self.d_prompt = m.get("q", ""), m.get("d", "")
        self.device = device or os.getenv("PAATTI_DEVICE") or _best_device()
        self.model = SentenceTransformer(m["id"], device=self.device, token=os.getenv("HF_TOKEN"),
                                         trust_remote_code=bool(m.get("remote")))
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = dim_fn()  # renamed in sentence-transformers 6

    def encode(self, texts, is_query):
        prompt = self.q_prompt if is_query else self.d_prompt
        texts = [prompt + t for t in texts] if prompt else list(texts)
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                            convert_to_numpy=True), dtype=np.float32)


class GeminiEmbedder(Embedder):
    """Gemini API embeddings: no model to host, fast cold start."""
    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768):
        from google import genai
        self.name = f"gemini:{model}"
        self.client = genai.Client(api_key=_api_key())
        self.model, self.dim = model, dim

    def encode(self, texts, is_query):
        from google.genai import types
        out = []
        for i in range(0, len(texts), 90):
            r = self.client.models.embed_content(
                model=self.model, contents=texts[i:i + 90],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.dim))
            out.extend(e.values for e in r.embeddings)
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


_KEY_NAMES = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY_1", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3")


def load_env() -> Optional[Path]:
    """Load the project's .env (v2/.env) no matter which directory the script runs from."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None
    for p in (Path(__file__).resolve().parent.parent / ".env", Path.cwd() / ".env"):
        if p.exists():
            load_dotenv(p, override=False)
            return p
    return None


def _api_key() -> str:
    env_file = load_env()
    keys = [os.getenv(k, "").strip().strip('"').strip("'") for k in _KEY_NAMES]
    keys = [k for k in keys if k]
    if not keys:
        where = f"{env_file} was loaded but has" if env_file else "no .env file found in the project folder, and the shell has"
        raise RuntimeError(f"Gemini key missing: {where} none of {', '.join(_KEY_NAMES)}. "
                           "Add a line like GOOGLE_API_KEY=your-key to v2/.env")
    import random
    return random.choice(keys)


def make_embedder(spec: Optional[str]) -> Optional[Embedder]:
    """spec: 'none' | 'gemini' | 'gemini:<model>' | registry key (e.g. 'bge-m3') | 'st:<hf_model_id_or_path>'"""
    from .embedders import MODELS
    if not spec or spec == "none":
        return None
    if spec.startswith("gemini"):
        return GeminiEmbedder(spec.split(":", 1)[1]) if ":" in spec else GeminiEmbedder()
    if spec.startswith("st:") or spec in MODELS:
        return STEmbedder(spec)
    raise ValueError(f"unknown embedder spec {spec}")


# ---------------- docs ----------------
def build_docs(kb: KnowledgeBase, mode: str = "v2") -> tuple[list[str], list[int]]:
    """Multi-vector docs: meaning+themes, plus one doc per situation."""
    texts, owners = [], []
    for v in kb.verses:
        texts.append(f"{v.literal} {v.meaning} Themes: {', '.join(v.themes)}. {v.translit} {v.tamil} {v.tamil_meaning}")
        owners.append(v.no)
        for s in v.situations:
            texts.append(s); owners.append(v.no)
    return texts, owners


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase, embedder: Optional[Embedder] = None, doc_mode: str = "v2",
                 cache_dir: Optional[Path] = None, use_bm25: bool = True, bm25_weight: Optional[float] = None):
        """bm25_weight: weight of the keyword signal in fusion relative to dense (1.0).
        Default: 0 when an embedder is present (benchmarks showed equal-weight fusion hurts strong
        dense models), keyword-only when there is no embedder."""
        self.kb, self.embedder, self.use_bm25 = kb, embedder, use_bm25
        if bm25_weight is None:
            bm25_weight = float(os.getenv("PAATTI_BM25_WEIGHT", "0")) if embedder is not None else 1.0
        self.bm25_weight = bm25_weight if use_bm25 else 0.0
        self.texts, owners = build_docs(kb, doc_mode)
        self.owners = np.asarray(owners)
        self.verse_ids = np.asarray(sorted(set(owners)))
        self.bm25 = BM25(self.texts) if (use_bm25 and (self.bm25_weight > 0 or embedder is None)) else None
        self.doc_vecs = None
        if embedder is not None:
            self.doc_vecs = self._load_or_encode(cache_dir, doc_mode)

    def _load_or_encode(self, cache_dir, doc_mode):
        h = hashlib.md5(("\n".join(self.texts) + self.embedder.name).encode()).hexdigest()[:12]
        cache_dir = Path(cache_dir or Path(__file__).resolve().parent.parent / ".cache")
        f = cache_dir / f"vecs_{doc_mode}_{h}.npy"
        if f.exists():
            return np.load(f)
        v = self.embedder.encode(self.texts, is_query=False)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(f, v)
        return v

    def _per_verse(self, doc_scores: np.ndarray) -> np.ndarray:
        out = np.full(len(self.verse_ids), -1e9)
        idx = {n: i for i, n in enumerate(self.verse_ids)}
        for s, n in zip(doc_scores, self.owners):
            j = idx[n]
            if s > out[j]:
                out[j] = s
        return out

    def search(self, query: str, k: int = 10, rrf_k: int = 60) -> list[tuple[int, float]]:
        rankings = []  # (scores, weight)
        if self.bm25 is not None:
            b = self.bm25.scores(query)
            if b.max() > 0:  # no keyword overlap (e.g. Tamil script) -> BM25 is pure noise, skip it
                rankings.append((self._per_verse(b), self.bm25_weight if self.doc_vecs is not None else 1.0))
        if self.doc_vecs is not None:
            q = self.embedder.encode([query], is_query=True)[0]
            rankings.append((self._per_verse(self.doc_vecs @ q), 1.0))
        if not rankings:
            if self.bm25 is not None:  # keyword-only and nothing matched
                rankings.append((self._per_verse(self.bm25.scores(query)), 1.0))
            else:
                raise RuntimeError("retriever has no signals")
        if len(rankings) == 1:
            s = rankings[0][0]
        else:  # weighted reciprocal rank fusion
            s = np.zeros(len(self.verse_ids))
            for r, w in rankings:
                order = np.argsort(-r)
                ranks = np.empty_like(order); ranks[order] = np.arange(len(order))
                s += w / (rrf_k + ranks + 1)
        top = np.argsort(-s)[:k]
        return [(int(self.verse_ids[i]), float(s[i])) for i in top]
