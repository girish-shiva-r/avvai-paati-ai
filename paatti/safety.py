"""Crisis detection. A life-advice bot must never answer a suicide/self-harm message
with a moral verse. These checks run before any retrieval or LLM call."""
import re

_PATTERNS = [
    r"\b(kill|end|take)\s+(my|my own)\s+life\b", r"\bsuicid", r"\bdon'?t\s+want\s+to\s+(live|be alive|exist)\b",
    r"\bwant\s+to\s+die\b", r"\bbetter\s+off\s+dead\b", r"\bself[\s-]?harm", r"\bcut(ting)?\s+myself\b",
    r"\bhurt\s+myself\b", r"\bno\s+reason\s+to\s+live\b", r"\bending\s+(it|my life)\b",
    r"saaganum", r"sethudalam", r"செத்துவிட", r"தற்கொலை", r"உயிரை\s*மாய்",
]
_RX = re.compile("|".join(_PATTERNS), re.I)

CRISIS_RESPONSE = (
    "Paatti is so glad you told me this. 🤍 What you are feeling is heavy, "
    "and you should not carry it alone.\n\n"
    "Please talk to someone right now:\n"
    "- **Tele-MANAS (India, free, 24×7):** call **14416** or **1-800-891-4416**\n"
    "- **Emergency:** **112**\n"
    "- Or reach out to someone you trust and tell them how you feel today.\n\n"
    "If you are outside India, please contact your local emergency number or a crisis line. "
    "I am still here. Would you like to tell me what has been weighing on you?"
)


def is_crisis(text: str) -> bool:
    return bool(_RX.search(text or ""))
