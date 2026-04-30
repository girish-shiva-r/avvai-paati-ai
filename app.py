import streamlit as st
import requests
import time

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
       DARK BOTTOM BAR  ← your requested change
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
       SIDEBAR COLLAPSE BUTTON (top-left white square)
       Using maximum specificity + all possible selectors
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
        // Try multiple selectors Streamlit uses across versions
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
                // Style inner SVG strokes
                btn.querySelectorAll('svg, path, polyline, line').forEach(el => {
                    el.style.setProperty('fill', '#FFFFFF', 'important');
                    el.style.setProperty('stroke', '#FFFFFF', 'important');
                });
            });
        });
    }
    // Run on load and observe DOM changes (Streamlit re-renders)
    styleCollapseBtn();
    const observer = new MutationObserver(styleCollapseBtn);
    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

# ── Backend URL ───────────────────────────────────────────────────
API_URL = st.secrets.get("API_URL", "https://GSR-608001-avvaiyar-brain.hf.space/chat")

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="paatti-header">
    <h1>👵 Avvaiyar Paatti AI</h1>
    <div class="tamil-sub">ஔவையார் பாட்டி</div>
    <div class="tagline">Ancient Tamil wisdom for modern life — ask me anything</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='color:#7A3B00; font-size:1rem; margin-bottom:0.5rem;'>💬 Ask Paatti about...</h3>", unsafe_allow_html=True)
    examples = [
        "I keep procrastinating my work",
        "My friend betrayed my trust",
        "I am losing motivation to study",
        "How do I deal with an arrogant person?",
        "I want to succeed but feel lost",
        "I feel lazy and unmotivated",
        "I get angry very easily",
        "I lied to someone I care about",
        "I feel jealous of others' success",
        "I never appreciate what I have",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()
    st.markdown("""
    <div style='font-size:0.78rem; color:#5A3000; line-height:1.6;'>
    <b>About</b><br>
    Avvaiyar Paatti gives life advice rooted in the 2,000-year-old Tamil text <i>Aathichoodi</i> by poet Avvaiyar.<br><br>
    <b>Stack</b><br>
    Fine-tuned Embeddings · FAISS · Gemini 2.0 Flash · FastAPI · Streamlit
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "வணக்கம் Kanna! 🙏\n\nI am Avvaiyar Paatti. \n\nThe ancient wisdom of Aathichoodi has guided hearts for 2,000 years. What is on your mind today?"
    })

for msg in st.session_state.messages:
    avatar = "👵" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Handle sidebar button prefill ────────────────────────────────
prefill = st.session_state.pop("prefill", None)

# ── Chat input ────────────────────────────────────────────────────
user_input = st.chat_input("Tell Paatti what's on your mind...") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="👵"):
        placeholder = st.empty()
        with st.spinner("Paatti is thinking... 🤔"):
            try:
                resp = requests.post(
                    API_URL,
                    json={"query": user_input},
                    timeout=60,
                )
                if resp.status_code == 200:
                    bot_text = resp.json().get("response", "")

                    displayed = ""
                    for word in bot_text.split():
                        displayed += word + " "
                        placeholder.markdown(displayed + "▌")
                        time.sleep(0.03)
                    placeholder.markdown(displayed.strip())

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": displayed.strip(),
                    })

                elif resp.status_code == 503:
                    placeholder.warning("Paatti is waking up — please send your message again in a few seconds 🙏")
                else:
                    placeholder.error(f"Something went wrong (error {resp.status_code}). Please try again.")

            except requests.exceptions.Timeout:
                placeholder.error("Took too long. Please try again — Paatti is still here 🙏")
            except requests.exceptions.ConnectionError:
                placeholder.error("Cannot reach the backend. Please check if the HF Space is running.")
            except Exception as e:
                placeholder.error(f"Unexpected error: {str(e)}")
