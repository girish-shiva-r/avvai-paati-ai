"""Avvaiyar Paatti engine: safety -> lookup -> hybrid retrieval -> LLM selection+answer -> validation."""
from __future__ import annotations
import logging, os, re, time
from typing import Optional

from .kb import KnowledgeBase
from .llm import LLM, PaattiReply, make_llm
from .retrieval import HybridRetriever, make_embedder
from .safety import CRISIS_RESPONSE, is_crisis

log = logging.getLogger("paatti")

SYSTEM = """You are Avvaiyar Paatti (ஔவையார் பாட்டி), the beloved Tamil poet Avvaiyar speaking as a warm, wise grandmother.
You guide your grandchild using ONLY the 109 verses of Aathichoodi (ஆத்திசூடி).

How to answer:
1. Decide the intent:
   - "advice": the grandchild shares a problem, feeling, dilemma or question about life.
   - "lookup": they ask what a specific verse means (a forced verse may be given to you).
   - "about": questions about Aathichoodi or Avvaiyar themselves (how many verses, who wrote it, how it is
     arranged, its history). Answer briefly and accurately; verse_no null unless they ask for an example.
   - "smalltalk": greetings, thanks, or chit-chat with no problem yet.
   - "offtopic": requests unrelated to life guidance (code, news, weather, trivia).
2. For advice, pick the ONE verse whose teaching most directly addresses the root of their situation.
   Read the CANDIDATE VERSES first (retrieved for this message); you may pick any verse from the FULL LIST
   if it fits clearly better. Think about the underlying issue (e.g. "my friend shared my secret" is about
   trusting the wrong people, not about speech). If they ask for "another verse", choose a different one
   from the ones already used in this conversation.
3. Write "response" as Paatti:
   - Do NOT use pet names or endearments of any kind, in any language: no "en chellame", "kanna",
     "thangame", "my dear", "my child", "dear one", "beta", "mere bache", "mone", "mole", etc. Begin by responding directly to what they said, acknowledging their specific
     situation or feeling in one line. Warmth comes from tone, not from labels.
   - If they are hurting (lonely, misunderstood, ashamed, sad), acknowledge the feeling first and choose a
     verse that offers them something constructive to do; never one that implies the pain is their fault.
     For loneliness or feeling unloved, do NOT pick verses about duties to parents, spouse or elders (their
     pain may come from exactly those people) unless they speak warmly of them; prefer verses about finding
     good company, belonging and community (e.g. 104, 45, 19, 37).
   - Explain what the verse teaches in simple words, faithful to its meaning (never invent meanings).
     Refer to it by its transliteration (e.g. "Aaruvadhu sinam"); do NOT write the Tamil script verse text,
     the app displays the verse card above your words.
   - Give 1-2 concrete, practical steps for today that follow directly from the verse's teaching.
     End with one short, encouraging line (no pet names).
   - 4-7 sentences. Warm, simple, not preachy. At most 2 emojis.
   - LANGUAGE: if they ask for a language by name ("in Hindi", "Malayalam la sollu", "en français"), reply
     fully in that language written in ITS OWN SCRIPT: Hindi/Marathi -> Devanagari (हिंदी), Malayalam -> മലയാളം,
     Telugu -> తెలుగు, Kannada -> ಕನ್ನಡ, Bengali -> বাংলা, Urdu -> اردو, Tamil -> தமிழ். Never answer a named-language
     request in Latin letters unless they explicitly ask for romanized/Hinglish/Tanglish.
     Otherwise reply in the language and script they wrote in, for any language. Only when they themselves
     write an Indian language in Latin letters (Tanglish, Hinglish) reply in that same romanized style. Keep the verse's transliteration (e.g. "Aaruvadhu sinam")
     as is, and explain its meaning in the reply language. Keep following the language they chose in later
     turns until they switch.
   - For serious medical, legal, financial or safety matters, also gently suggest talking to a doctor,
     counsellor, lawyer or trusted elder.
4. For "about": no practical steps needed; 2-4 warm sentences, then invite a life question.
   For "smalltalk": greet warmly, invite them to share what is on their mind; verse_no may be null.
   For "offtopic": kindly say Paatti only shares Aathichoodi's wisdom for life, and invite a life question; verse_no null.
5. "reason": one short sentence on why this verse fits (for logging, not shown to the user).
Return JSON only."""


# Pet names Paatti must not use. "Strong" ones are unambiguous and are stripped even without a comma;
# short ambiguous words (beta, darling, mone...) only when set off by a comma.
_STRONG = (r"(?:en\s+)?(?:chellame|chellam|thangame|my\s+dear(?:est)?|my\s+child|dear\s+one|dear\s+child|my\s+darling|"
           r"என்\s*செல்லமே|செல்லமே|தங்கமே|"
           r"mere\s+(?:pyare\s+|pyaare\s+)?bacc?h?(?:e|on|a)|मेरे\s+(?:प्यारे\s+)?बच्च(?:े|ों)|"
           r"മോനേ|മോളേ)")
_WEAK = r"(?:kanna|kannu|கண்ணா|beta|beti|बेटा|बेटी|bachcha|baccha|darling|mone|mole)"
_PET = r"(?:" + _STRONG + r"|" + _WEAK + r")"
_END = r"(?=\s*[.!?।]?\s*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF]\s*)*$)"
_LEAD = re.compile(r"^\s*(?:oh\s+)?" + _PET + r"\s*[,!.:।]+\s*", re.I)
_TAIL = re.compile(r"(?:,\s*" + _PET + r"|[,\s]\s*" + _STRONG + r")\s*" + _END, re.I)


def strip_endearments(text: str) -> str:
    """Safety net in case the model still opens/closes with a pet name the prompt asks it to avoid."""
    t = (text or "").strip()
    t2 = _LEAD.sub("", t)
    if t2 != t and t2:
        t2 = t2[0].upper() + t2[1:]
    return _TAIL.sub("", t2).strip()


class Paatti:
    def __init__(self, kb: Optional[KnowledgeBase] = None, llm: Optional[LLM] = None,
                 embedder_spec: Optional[str] = None, k: int = 8, history_turns: int = 6):
        self.kb = kb or KnowledgeBase()
        self.llm = llm or make_llm()
        spec = embedder_spec if embedder_spec is not None else os.getenv("PAATTI_EMBEDDER", "mpnet-ml")
        try:
            emb = make_embedder(spec)
        except Exception as e:  # never fail startup because of embeddings: BM25 still works
            log.warning("embedder %s unavailable (%s); using BM25 only", spec, e)
            emb = None
        self.retriever = HybridRetriever(self.kb, emb)
        self.k, self.history_turns = k, history_turns
        self.full_list = "\n".join(v.compact() for v in self.kb.verses)

    # ----------------------------------------------------------
    def _retrieval_query(self, query: str, history: list[dict]) -> str:
        prev = [t["content"] for t in history if t["role"] == "user"][-1:]
        if prev and len(query.split()) < 8:  # short follow-ups need context
            return prev[0] + " " + query
        return query

    def answer(self, query: str, history: Optional[list[dict]] = None) -> dict:
        t0 = time.time()
        history = (history or [])[-2 * self.history_turns:]
        query = (query or "").strip()
        if not query:
            return {"response": "Tell Paatti what is on your mind.", "verse": None, "intent": "smalltalk"}

        if is_crisis(query):
            return {"response": CRISIS_RESPONSE, "verse": None, "intent": "crisis", "latency": time.time() - t0}

        forced = self.kb.lookup(query)
        try:
            hits = self.retriever.search(self._retrieval_query(query, history), k=self.k)
        except Exception as e:
            log.warning("retrieval failed: %s", e)
            hits = []
        cand_nos = [n for n, _ in hits]
        if forced and forced not in cand_nos:
            cand_nos = [forced] + cand_nos
        used = [t.get("verse_no") for t in history if t.get("verse_no")]

        user_block = (
            (f"FORCED VERSE (the user is asking about this exact verse; intent=lookup, verse_no={forced}).\n\n" if forced else "")
            + "CANDIDATE VERSES (retrieved, best first):\n"
            + "\n".join(self.kb.get(n).prompt_block() for n in cand_nos)
            + f"\n\nFULL LIST OF 109 VERSES:\n{self.full_list}\n\n"
            + (f"Verses already used in this conversation: {used}\n\n" if used else "")
            + f"GRANDCHILD'S MESSAGE:\n{query}"
        )
        turns = [{"role": t["role"], "content": t["content"]} for t in history] + [{"role": "user", "content": user_block}]

        try:
            rep: PaattiReply = self.llm.reply(SYSTEM, turns)
        except Exception as e:
            log.exception("LLM failed")
            v = self.kb.get(forced or (cand_nos[0] if cand_nos else 1))
            return {"response": f"Paatti is a little tired right now, but remember this: {v.meaning}",
                    "verse": v.card(), "verse_no": v.no, "intent": "fallback", "error": str(e)[:300],
                    "candidates": cand_nos, "latency": time.time() - t0}

        # ---- validation: verse text always comes from the KB, never from the LLM ----
        # A verse card is shown only when a verse is actually being discussed:
        #  - lookup: always the verse the user named
        #  - advice: the LLM's pick, or the top retrieved verse if it returned none/invalid
        #  - about/lookup-by-LLM: only if the LLM chose a valid verse (e.g. "give an example")
        #  - smalltalk/offtopic: never
        valid = rep.verse_no if (rep.verse_no is not None and self.kb.get(rep.verse_no)) else None
        if forced:
            vno = forced
        elif rep.intent == "advice":
            vno = valid or (cand_nos[0] if cand_nos else None)
        elif rep.intent in ("about", "lookup"):
            vno = valid
        else:
            vno = None
        verse = self.kb.get(vno) if vno else None
        return {"response": strip_endearments(rep.response), "verse": verse.card() if verse else None,
                "verse_no": vno, "intent": "lookup" if forced else rep.intent, "reason": rep.reason,
                "candidates": cand_nos, "latency": round(time.time() - t0, 2)}
