"""Streamlit UI for Avvaiyar Paatti. Set API_URL to use the hosted backend; leave empty to run the engine in-process."""
import io, json, os, time
import requests
import streamlit as st

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Avvaiyar Paatti AI",
    page_icon="👵",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif:ital,wght@0,400;0,700;1,400&display=swap');

    /* ══════════════════════════════════════════════
       BASE PAGE
    ══════════════════════════════════════════════ */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stHeader"] > div {
        background-color: #FDF6EC !important;
    }

    /* ══════════════════════════════════════════════
       DARK BOTTOM BAR
    ══════════════════════════════════════════════ */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div,
    [data-testid="stBottom"] > div > div > div {
        background-color: #1A0A00 !important;
        border-top: 2px solid #C47A2B !important;
    }

    /* Chat input inside dark bar */
    [data-testid="stChatInput"] {
        background-color: transparent !important;
        padding: 0.6rem 1rem !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #2D1200 !important;
        border: 1.5px solid #C47A2B !important;
        border-radius: 24px !important;
        color: #FFE8C0 !important;
        caret-color: #FFE8C0 !important;
        font-size: 0.97rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #A07040 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #E07B00 !important;
        box-shadow: 0 0 0 2px rgba(196, 122, 43, 0.3) !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #C47A2B !important;
        border-radius: 50% !important;
        border: none !important;
    }
    [data-testid="stChatInput"] button svg,
    [data-testid="stChatInput"] button path {
        fill: #FFFFFF !important;
        stroke: none !important;
    }

    /* ══════════════════════════════════════════════
       SIDEBAR COLLAPSE BUTTON
    ══════════════════════════════════════════════ */
    div[data-testid="stSidebarCollapseButton"] button,
    div[data-testid="stSidebarCollapseButton"] > div > button,
    section[data-testid="stSidebar"] ~ div button,
    button[title="Collapse sidebar"],
    button[aria-label="Collapse sidebar"],
    button[title="Open sidebar"],
    button[aria-label="Open sidebar"],
    [data-testid="collapsedControl"] button {
        background-color: #C47A2B !important;
        border: 2px solid #A05A10 !important;
        border-radius: 8px !important;
        width: 36px !important;
        height: 36px !important;
        padding: 4px !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="collapsedControl"] button:hover {
        background-color: #A05A10 !important;
    }
    div[data-testid="stSidebarCollapseButton"] svg,
    div[data-testid="stSidebarCollapseButton"] path,
    div[data-testid="stSidebarCollapseButton"] polyline,
    div[data-testid="stSidebarCollapseButton"] line,
    [data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] path,
    [data-testid="collapsedControl"] polyline,
    [data-testid="collapsedControl"] line {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
        color: #FFFFFF !important;
    }

    /* ══════════════════════════════════════════════
       TOP-RIGHT HEADER BUTTONS
    ══════════════════════════════════════════════ */
    header[data-testid="stHeader"] button,
    [data-testid="stToolbarActions"] button {
        background-color: #C47A2B !important;
        border: 1.5px solid #A05A10 !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        opacity: 1 !important;
    }
    header[data-testid="stHeader"] button:hover {
        background-color: #A05A10 !important;
    }
    header[data-testid="stHeader"] button svg,
    header[data-testid="stHeader"] button path,
    [data-testid="stToolbarActions"] svg,
    [data-testid="stToolbarActions"] path {
        fill: #FFFFFF !important;
        stroke: none !important;
    }

    /* ══════════════════════════════════════════════
       BOTTOM-RIGHT FLOATING ICONS
    ══════════════════════════════════════════════ */
    [data-testid="stStatusWidget"],
    [data-testid="stStatusWidget"] > div {
        background-color: #C47A2B !important;
        border-radius: 20px !important;
        border: 1.5px solid #A05A10 !important;
    }
    [data-testid="stStatusWidget"] svg,
    [data-testid="stStatusWidget"] path {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }
    [class*="viewerBadge"],
    [class*="profileContainer"] {
        background-color: #C47A2B !important;
        border-radius: 50% !important;
        border: 2px solid #A05A10 !important;
    }
    [class*="profileContainer"] svg,
    [class*="profileContainer"] path {
        fill: #FFFFFF !important;
    }

    /* ══════════════════════════════════════════════
       SIDEBAR
    ══════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background-color: #FFF3DC !important;
        border-right: 2px solid #E8C97A;
    }
    [data-testid="stSidebar"] .stButton > button {
        background-color: #8B2500 !important;
        color: #FFFFFF !important;
        border: 1.5px solid #6B1A00 !important;
        border-radius: 10px !important;
        text-align: left !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 0.8rem !important;
        width: 100% !important;
        margin-bottom: 3px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
        transition: background-color 0.18s, transform 0.1s !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #6B1A00 !important;
        transform: translateX(2px) !important;
    }
    [data-testid="stSidebar"] .stButton:last-of-type > button {
        background-color: #5C0F0F !important;
        border-color: #3D0808 !important;
    }
    [data-testid="stSidebar"] .stButton:last-of-type > button:hover {
        background-color: #3D0808 !important;
    }

    /* ══════════════════════════════════════════════
       CHAT BUBBLES
    ══════════════════════════════════════════════ */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #FFF8EE !important;
        border: 1px solid #F0D9A0 !important;
        border-radius: 14px !important;
        padding: 0.8rem 1rem !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #EEF2FF !important;
        border: 1px solid #C5CFF0 !important;
        border-radius: 14px !important;
        padding: 0.8rem 1rem !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stChatMessageContent"] p {
        font-size: 1rem;
        line-height: 1.65;
        color: #2C1A00;
    }

    /* ── Page header ── */
    .paatti-header {
        text-align: center;
        padding: 1.5rem 0 0.8rem 0;
    }
    .paatti-header h1 {
        font-family: 'Noto Serif', Georgia, serif;
        color: #7A3B00;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .paatti-header .tamil-sub {
        font-size: 1.15rem;
        color: #C47A2B;
        font-style: italic;
        margin-bottom: 0.3rem;
    }
    .paatti-header .tagline {
        font-size: 0.92rem;
        color: #8B6343;
    }
    hr {
        border: none;
        border-top: 1px solid #E8C97A;
        margin: 0.5rem 0 1rem 0;
    }
    .stSpinner > div {
        border-top-color: #E07B00 !important;
    }
</style>

<!-- JS fallback: forcibly style the sidebar collapse button by finding it in the DOM -->
<script>
(function() {
    function styleCollapseBtn() {
        const selectors = [
            '[data-testid="stSidebarCollapseButton"] button',
            '[data-testid="collapsedControl"] button',
            'button[title*="sidebar"]',
            'button[aria-label*="sidebar"]',
        ];
        selectors.forEach(sel => {
            document.querySelectorAll(sel).forEach(btn => {
                btn.style.setProperty('background-color', '#C47A2B', 'important');
                btn.style.setProperty('border', '2px solid #A05A10', 'important');
                btn.style.setProperty('border-radius', '8px', 'important');
                btn.style.setProperty('color', '#fff', 'important');
                btn.querySelectorAll('svg, path, polyline, line').forEach(el => {
                    el.style.setProperty('fill', '#FFFFFF', 'important');
                    el.style.setProperty('stroke', '#FFFFFF', 'important');
                });
            });
        });
    }
    styleCollapseBtn();
    const observer = new MutationObserver(styleCollapseBtn);
    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.verse-card{background:linear-gradient(135deg,#7A3B00,#A0522D);color:#FFF3DC;border-radius:14px;
  padding:.9rem 1.1rem;margin:.2rem 0 .7rem 0;border:1px solid #C47A2B}
.verse-card .vno{font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;opacity:.85}
.verse-card .ta{font-family:'Noto Serif Tamil','Noto Serif',serif;font-size:1.35rem;font-weight:700;margin:.15rem 0}
.verse-card .tr{font-style:italic;opacity:.92}
.verse-card .lit{font-size:.9rem;margin-top:.3rem;opacity:.95}
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


def ask(query, history, audio=None):
    hist = [{"role": m["role"], "content": m["content"], "verse_no": m.get("verse_no")} for m in history
            if m.get("role") in ("user", "assistant")]
    if API_URL:
        base = API_URL.rstrip("/").removesuffix("/chat")
        if audio is not None:
            r = requests.post(f"{base}/chat_audio", files={"file": (audio.name, audio.getvalue(), "audio/wav")},
                              data={"history": json.dumps(hist)}, timeout=90)
        else:
            r = requests.post(f"{base}/chat", json={"query": query, "history": hist}, timeout=90)
        if r.status_code == 503:
            raise RuntimeError("Paatti is waking up — please send again in a few seconds 🙏")
        r.raise_for_status()
        return r.json()
    bot = local_bot()
    if audio is not None:
        text = bot.llm.transcribe(audio.getvalue(), "audio/wav")
        out = bot.answer(text, hist); out["transcribed_text"] = text
        return out
    return bot.answer(query, hist)


def verse_card_html(v):
    return (f"<div class='verse-card'><div class='vno'>Aathichoodi · {v['no']}</div>"
            f"<div class='ta'>{v['tamil']}</div><div class='tr'>{v['translit']}</div>"
            f"<div class='lit'>“{v['literal']}”</div></div>")


def speak(text):
    try:
        from gtts import gTTS
        fp = io.BytesIO(); gTTS(text=text, lang="en", tld="co.in").write_to_fp(fp); fp.seek(0)
        st.audio(fp, format="audio/mp3", autoplay=True)
    except Exception as e:
        st.caption(f"(voice unavailable: {e})")


# ── Header ──
st.markdown("""
<div class="paatti-header">
    <h1>👵 Avvaiyar Paatti AI</h1>
    <div class="tamil-sub">ஔவையார் பாட்டி</div>
    <div class="tagline">Ancient Tamil wisdom for modern life — ask me anything</div>
</div>""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ──
with st.sidebar:
    st.markdown("<h3 style='color:#7A3B00;font-size:1rem;margin-bottom:.5rem;'>💬 Ask Paatti about...</h3>", unsafe_allow_html=True)
    for ex in ["I keep procrastinating my work", "My friend betrayed my trust", "I am losing motivation to study",
               "I get angry very easily", "I spend my whole salary every month", "I feel jealous of others' success",
               "enakku romba kovam varudhu", "What does verse 54 mean?"]:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["prefill"] = ex
    st.divider()
    voice_on = st.toggle("🔊 Paatti's voice", value=False)
    show_debug = st.toggle("🔎 Show retrieval details", value=False)
    st.markdown("""<div style='font-size:.78rem;color:#5A3000;line-height:1.6;'><b>About</b><br>
    Life advice rooted in the <i>Aathichoodi</i> of Avvaiyar (109 verses).<br><br>
    <b>Stack</b><br>Curated verse KB · Hybrid BM25 + embeddings · Gemini · FastAPI · Streamlit</div>""",
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
        st.markdown("வணக்கம் Kanna! 🙏\n\nI am Avvaiyar Paatti. Tell me what is on your mind, "
                    "in English, தமிழ், or Tanglish, and I will share a verse of Aathichoodi to guide you.")

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="👵" if m["role"] == "assistant" else "🧑"):
        if m.get("verse"):
            st.markdown(verse_card_html(m["verse"]), unsafe_allow_html=True)
        st.markdown(m["content"])

prefill = st.session_state.pop("prefill", None)
audio_value = st.audio_input("🎙️ Speak to Paatti")
user_input = st.chat_input("Type to Paatti…") or prefill

if user_input or audio_value:
    if audio_value and st.session_state.get("last_audio") == audio_value.getvalue()[:2048] and not user_input:
        st.stop()  # don't resend the same recording on rerun
    shown = user_input or "🎤 (voice message)"
    with st.chat_message("user", avatar="🧑"):
        st.markdown(shown)
    with st.chat_message("assistant", avatar="👵"):
        with st.spinner("Paatti is thinking… 🌿"):
            try:
                out = ask(user_input, st.session_state.messages, None if user_input else audio_value)
            except Exception as e:
                st.error(str(e)); st.stop()
        if not user_input:
            st.session_state["last_audio"] = audio_value.getvalue()[:2048]
            shown = f"🎤 *{out.get('transcribed_text', '')}*"
        if out.get("verse"):
            st.markdown(verse_card_html(out["verse"]), unsafe_allow_html=True)
        placeholder, typed = st.empty(), ""
        for w in out["response"].split(" "):
            typed += w + " "; placeholder.markdown(typed + "▌"); time.sleep(0.015)
        placeholder.markdown(out["response"])
        if show_debug:
            st.caption(f"intent={out.get('intent')} · verse={out.get('verse_no')} · candidates={out.get('candidates')} "
                       f"· {out.get('latency')}s · {out.get('reason', '')}")
        if voice_on:
            speak(out["response"])
    st.session_state.messages.append({"role": "user", "content": shown.replace("🎤 ", "").strip("*") if not user_input else shown})
    st.session_state.messages.append({"role": "assistant", "content": out["response"], "verse": out.get("verse"),
                                      "verse_no": out.get("verse_no")})
