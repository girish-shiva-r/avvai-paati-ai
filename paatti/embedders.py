"""Registry of RAW (not fine-tuned) multilingual embedding models with the exact
query/document prompts each model card prescribes. Wrong or missing prefixes
silently cost several points of retrieval accuracy, so they live in one place.

Fine-tuned checkpoints (e.g. GSR-608001/*) are deliberately NOT in here: they were
trained on the v1 documents and would leak/skew the comparison.
"""
from __future__ import annotations

TASK = "Given a person's description of a life problem, retrieve the proverb that gives the most fitting advice"

# key -> dict(id, q (query prompt), d (doc prompt), remote (trust_remote_code), gated, note, group)
MODELS = {
    # --- small / fast (deployable on a free CPU Space) ---
    "e5-small":   dict(id="intfloat/multilingual-e5-small", q="query: ", d="passage: ", group="core", note="118M, 384-d"),
    "e5-base":    dict(id="intfloat/multilingual-e5-base", q="query: ", d="passage: ", group="core", note="278M, 768-d"),
    "mpnet-ml":   dict(id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", q="", d="", group="core", note="278M, paraphrase-trained"),
    "labse":      dict(id="sentence-transformers/LaBSE", q="", d="", group="extra", note="471M, translation pairs, weak for retrieval"),
    "l3cube-ta":  dict(id="l3cube-pune/tamil-sentence-similarity-sbert", q="", d="", group="core", note="Tamil SBERT used in the original Archive project"),
    "gte-ml":     dict(id="Alibaba-NLP/gte-multilingual-base", q="", d="", remote=True, group="core", note="305M, 768-d"),
    "muril-st":   dict(id="Yunika/muril-sentence-transformer", q="", d="", group="core", note="MuRIL (Indic BERT), QQP-trained"),
    "vyakyarth":  dict(id="krutrim-ai-labs/Vyakyarth", q="", d="", group="core", note="Indic sentence model (Krutrim licence)"),
    # --- large ---
    "e5-large":   dict(id="intfloat/multilingual-e5-large", q="query: ", d="passage: ", group="core", note="560M, base of aathichoodi_pro_model"),
    "e5-large-instruct": dict(id="intfloat/multilingual-e5-large-instruct", q=f"Instruct: {TASK}\nQuery: ", d="", group="core", note="560M, instruction-tuned"),
    "e5-large-v2": dict(id="intfloat/e5-large-v2", q="query: ", d="passage: ", group="core", note="English-only; most queries are English"),
    "bge-m3":     dict(id="BAAI/bge-m3", q="", d="", group="core", note="568M, strong multilingual incl. Tamil"),
    "arctic-l2":  dict(id="Snowflake/snowflake-arctic-embed-l-v2.0", q="query: ", d="", group="core", note="568M"),
    "qwen3-0.6b": dict(id="Qwen/Qwen3-Embedding-0.6B", q=f"Instruct: {TASK}\nQuery:", d="", group="core", note="600M, needs transformers>=4.51"),
    # --- optional: gated or restrictive licence ---
    "gemma-300m": dict(id="google/embeddinggemma-300m", q="task: search result | query: ", d="title: none | text: ",
                       gated=True, group="extra", note="gated: accept licence on HF first"),
    "jina-v3":    dict(id="jinaai/jina-embeddings-v3", q="Represent the query for retrieving evidence documents: ",
                       d="Represent the document for retrieval: ", remote=True, group="extra", note="CC BY-NC licence"),
}


def resolve(spec: str) -> dict:
    """'bge-m3' (registry key) or 'st:<any hf id>' (no prompts)."""
    if spec in MODELS:
        return {"key": spec, **MODELS[spec]}
    if spec.startswith("st:"):
        mid = spec[3:]
        for k, m in MODELS.items():
            if m["id"] == mid:
                return {"key": k, **m}
        return {"key": mid, "id": mid, "q": "", "d": "", "note": "unregistered: no prompts applied"}
    raise KeyError(spec)
