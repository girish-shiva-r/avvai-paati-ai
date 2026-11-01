"""Voice helpers shared by the app and tests."""
import re

_TTS_SCRIPTS = [("ta", 0x0B80, 0x0BFF), ("hi", 0x0900, 0x097F), ("ml", 0x0D00, 0x0D7F), ("te", 0x0C00, 0x0C7F),
                ("kn", 0x0C80, 0x0CFF), ("bn", 0x0980, 0x09FF), ("ur", 0x0600, 0x06FF)]


def tts_lang(text: str) -> str:
    """gTTS language code for the dominant script of a reply. Latin-script text (English, Tanglish,
    Hinglish) -> "en" (spoken with an Indian-English accent by the app)."""
    letters = [ord(c) for c in text or "" if c.isalpha()]
    best, n = "en", 0
    for code, lo, hi in _TTS_SCRIPTS:
        k = sum(lo <= c <= hi for c in letters)
        if k > n:
            best, n = code, k
    return best if letters and n / len(letters) > 0.3 else "en"


# Things speech models return for silence/noise instead of an empty string.
_NON_SPEECH = re.compile(
    r"^\W*(?:\[[^\]]*\]|\([^)]*\)|<[^>]*>|\*[^*]*\*|no\s+(?:intelligible\s+)?speech(?:\s+detected)?|silence|inaudible|"
    r"unintelligible|empty(?:\s+string)?|none|null|n/?a|music|background\s+noise)\W*$", re.I)


def clean_transcript(text: str) -> str:
    """Return the transcript, or "" if it contains no real words (e.g. "00:00", "[silence]", "...")."""
    t = (text or "").strip().strip('"').strip()
    if not any(c.isalpha() for c in t):  # only digits / punctuation / timestamps
        return ""
    if _NON_SPEECH.match(t):
        return ""
    return t
