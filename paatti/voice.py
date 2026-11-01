"""Voice-input helpers (speech-to-text cleanup) shared by the app, API and tests."""
import re

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
