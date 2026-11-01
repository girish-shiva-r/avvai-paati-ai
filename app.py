"""Streamlit UI for Avvaiyar Paatti. Set API_URL to use the hosted backend; leave empty to run the engine in-process."""
import hashlib, html, json, os, time
import requests
import streamlit as st


# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Avvaiyar Paatti AI",
    page_icon="👵",
    layout="centered",
    initial_sidebar_state="auto",
)

# ── Styling ───────────────────────────────────────────────────────
# Colours come from .streamlit/config.toml (light theme). This CSS only adds the warm accents.
# Palette (WCAG contrast on its background): text #2A1708 (15.9:1), muted #5C4330 (8.5:1),
# maroon #8C2A0C (8.0:1), verse card #4A1508 with ivory #FFF6E6 (14:1) and gold #F6CF6A (10:1).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif:wght@600;700&family=Noto+Serif+Tamil:wght@500;700&family=Noto+Sans+Tamil:wght@400;600&display=swap');

:root{
  --ink:#2A1708; --muted:#5C4330; --maroon:#8C2A0C; --maroon-dk:#6E1F08;
  --ivory:#FBF6EE; --sand:#F3E6D0; --line:#E2CDA6; --gold:#F6CF6A;
  --card:#4A1508; --card-text:#FFF6E6; --user:#E9EEF8; --user-ink:#1C2740;
}
/* Tamil text everywhere gets a proper Tamil font; icons keep Streamlit's icon font */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{
  font-family:"Source Sans Pro","Source Sans 3",'Noto Sans Tamil',sans-serif}

/* ---------- header ---------- */
.paatti-header{text-align:center;padding:.6rem 0 .2rem}
.paatti-header h1{font-family:'Noto Serif',Georgia,serif;color:var(--maroon);font-size:2.35rem;
  font-weight:700;margin:0;letter-spacing:-.5px}
.paatti-header .tamil-sub{font-family:'Noto Serif Tamil',serif;color:var(--ink);font-size:1.15rem;
  font-weight:500;margin:.25rem 0}
.paatti-header .tagline{color:var(--muted);font-size:.95rem}
.paatti-rule{height:3px;margin:.9rem auto 1.4rem;width:120px;border-radius:3px;
  background:linear-gradient(90deg,var(--maroon),var(--gold),var(--maroon))}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{border-right:1px solid var(--line)}
[data-testid="stSidebar"] h3{color:var(--maroon)!important;font-size:1rem!important}
[data-testid="stSidebar"] .stButton>button{
  background:#FFFFFF;color:var(--maroon);border:1.5px solid var(--maroon);border-radius:10px;
  font-weight:500;font-size:.88rem;text-align:left;justify-content:flex-start;padding:.45rem .8rem;
  transition:background .15s,color .15s}
[data-testid="stSidebar"] .stButton>button:hover,
[data-testid="stSidebar"] .stButton>button:focus-visible{background:var(--maroon);color:#FFFFFF;border-color:var(--maroon)}
[data-testid="stSidebar"] .stButton>button p{color:inherit}
.side-note{font-size:.82rem;color:var(--muted);line-height:1.6}
.side-note b{color:var(--ink)}

/* ---------- chat bubbles ---------- */
[data-testid="stChatMessage"]{border-radius:16px;padding:.85rem 1rem;margin-bottom:.7rem}
.mk{display:none}
[data-testid="stChatMessage"]:has(.mk-paatti){
  background:#FFFFFF;border:1px solid var(--line);box-shadow:0 1px 2px rgba(74,21,8,.06)}
[data-testid="stChatMessage"]:has(.mk-user){
  background:var(--user);border:1px solid #C9D3E8}
[data-testid="stChatMessage"]:has(.mk-user) p{color:var(--user-ink)}
[data-testid="stChatMessageContent"] p{font-size:1rem;line-height:1.7;color:var(--ink)}

/* ---------- verse card ---------- */
.verse-card{background:var(--card);color:var(--card-text);border-radius:14px;padding:1rem 1.15rem;
  margin:.15rem 0 .8rem;border-left:5px solid var(--gold)}
.verse-card .vno{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--gold);font-weight:600}
.verse-card .ta{font-family:'Noto Serif Tamil','Noto Serif',serif;font-size:1.45rem;font-weight:700;
  color:#FFFFFF;margin:.25rem 0 .1rem;line-height:1.5}
.verse-card .tr{font-style:italic;color:var(--gold)}
.verse-card .lit{font-size:.92rem;margin-top:.35rem;color:var(--card-text)}

/* ---------- inputs ---------- */
[data-testid="stChatInput"]>div{border:1.5px solid var(--maroon)!important;background:#FFFFFF!important;border-radius:14px}
[data-testid="stChatInput"] textarea{color:var(--ink)!important}
[data-testid="stChatInput"] textarea::placeholder{color:var(--muted)!important;opacity:1}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:#FFFFFF}
[data-testid="stExpander"] summary p{color:var(--maroon);font-weight:600}
.stSpinner p{color:var(--muted)}
</style>""", unsafe_allow_html=True)


def _secret(name, default=None):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


API_URL = _secret("API_URL")  # e.g. https://GSR-608001-avvaiyar-brain.hf.space  (empty -> run engine in-process)
for k in ("GOOGLE_API_KEY", "PAATTI_EMBEDDER", "PAATTI_LLM", "PAATTI_LLM_FALLBACK", "PAATTI_THINKING_LEVEL"):
    if _secret(k) and not os.getenv(k):
        os.environ[k] = _secret(k)


@st.cache_resource(show_spinner="Paatti is waking up…")
def local_bot():
    from paatti import Paatti
    return Paatti()


def ask(query, history):
    hist = [{"role": m["role"], "content": m["content"].removeprefix("🎤 "), "verse_no": m.get("verse_no")}
            for m in history if m.get("role") in ("user", "assistant")]
    if API_URL:
        base = API_URL.rstrip("/").removesuffix("/chat")
        r = requests.post(f"{base}/chat", json={"query": query, "history": hist}, timeout=90)
        if r.status_code == 503:
            raise RuntimeError("Paatti is waking up — please send again in a few seconds 🙏")
        r.raise_for_status()
        return r.json()
    return local_bot().answer(query, hist)


def transcribe(audio) -> str:
    """Speech -> text (any language). Returns "" when nothing intelligible was said."""
    data = audio.getvalue()
    if API_URL:
        base = API_URL.rstrip("/").removesuffix("/chat")
        r = requests.post(f"{base}/transcribe", files={"file": (audio.name, data, "audio/wav")}, timeout=90)
        r.raise_for_status()
        return r.json().get("text", "").strip()
    return local_bot().llm.transcribe(data, "audio/wav").strip()


def verse_card_html(v):
    return (f"<div class='verse-card'><div class='vno'>Aathichoodi · {v['no']}</div>"
            f"<div class='ta'>{v['tamil']}</div><div class='tr'>{v['translit']}</div>"
            f"<div class='lit'>“{v['literal']}”</div></div>")


def user_md(text):
    return "<span class='mk mk-user'></span>" + html.escape(text)


def paatti_md(text):
    return "<span class='mk mk-paatti'></span>" + text


# ── Header ──
st.markdown("""
<div class="paatti-header">
    <h1>👵 Avvaiyar Paatti AI</h1>
    <div class="tamil-sub">ஔவையார் பாட்டி</div>
    <div class="tagline">Ancient Tamil wisdom for modern life — ask me anything</div>
</div><div class="paatti-rule"></div>""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 💬 Ask Paatti about…")
    for ex in ["I keep procrastinating my work", "My friend betrayed my trust", "I am losing motivation to study",
               "I get angry very easily", "I spend my whole salary every month", "I feel jealous of others' success",
               "enakku romba kovam varudhu", "What does verse 54 mean?"]:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["prefill"] = ex
    st.divider()
    st.markdown("### 🎙️ Speak to Paatti")
    audio_value = st.audio_input("Record your message", label_visibility="collapsed")
    st.divider()
    show_debug = st.toggle("🔎 Show retrieval details", value=False)
    st.markdown("""<div class='side-note'><b>About</b><br>
    Life advice rooted in the <i>Aathichoodi</i> of Avvaiyar (109 verses).<br><br>
    <b>How it works</b><br>Curated verse knowledge base · multilingual embeddings · Gemini</div>""",
                unsafe_allow_html=True)
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── History ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="👵"):
        st.markdown(paatti_md("வணக்கம்! 🙏\n\nI am Avvaiyar Paatti. Tell me what is on your mind, "
                              "in any language you like, and I will share a verse of Aathichoodi to guide you."),
                    unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="👵" if m["role"] == "assistant" else "🧑"):
        if m.get("verse"):
            st.markdown(verse_card_html(m["verse"]), unsafe_allow_html=True)
        st.markdown(paatti_md(m["content"]) if m["role"] == "assistant" else user_md(m["content"]),
                    unsafe_allow_html=True)

prefill = st.session_state.pop("prefill", None)
user_input = st.chat_input("Type to Paatti…") or prefill

query, shown = user_input, user_input
if not user_input and audio_value is not None:
    digest = hashlib.sha1(audio_value.getvalue()).hexdigest()
    if st.session_state.get("last_audio") != digest:      # a new recording (not a rerun of the old one)
        st.session_state["last_audio"] = digest
        with st.spinner("Paatti is listening… 🎧"):
            try:
                heard = transcribe(audio_value)
            except Exception as e:
                heard = ""
                st.error(f"Paatti couldn't process the recording: {e}")
        if heard:
            query, shown = heard, f"🎤 {heard}"
        else:
            st.warning("Paatti couldn't hear any words in that recording. Please try again, a little closer to the mic.")

if query:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_md(shown), unsafe_allow_html=True)
    with st.chat_message("assistant", avatar="👵"):
        with st.spinner("Paatti is thinking… 🌿"):
            try:
                out = ask(query, st.session_state.messages)
            except Exception as e:
                st.error(str(e)); st.stop()
        if out.get("verse"):
            st.markdown(verse_card_html(out["verse"]), unsafe_allow_html=True)
        placeholder, typed = st.empty(), ""
        for w in out["response"].split(" "):
            typed += w + " "; placeholder.markdown(paatti_md(typed + "▌"), unsafe_allow_html=True); time.sleep(0.015)
        placeholder.markdown(paatti_md(out["response"]), unsafe_allow_html=True)
        if show_debug:
            st.caption(f"intent={out.get('intent')} · verse={out.get('verse_no')} · candidates={out.get('candidates')} "
                       f"· {out.get('latency')}s · {out.get('reason', '')}")
    st.session_state.messages.append({"role": "user", "content": shown})
    st.session_state.messages.append({"role": "assistant", "content": out["response"], "verse": out.get("verse"),
                                      "verse_no": out.get("verse_no")})
