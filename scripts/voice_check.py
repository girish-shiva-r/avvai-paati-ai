"""Voice test suite: speech -> transcript -> Paatti reply, with your real Gemini key, PASS/FAIL per case.

  python3 scripts/voice_check.py --suite            # macOS: generate clips with `say`, run all, print a scorecard
  python3 scripts/voice_check.py my_clip.m4a ...    # just transcribe + answer your own recordings

Clips are made with the voices installed on your Mac (picked by locale: en_IN, hi_IN, ta_IN, ml_IN, te_IN...).
Cases whose language has no installed voice are skipped (add voices: System Settings > Accessibility >
Spoken Content > System voice > Manage Voices). Each case uses ~2 Gemini requests.
Supported formats for your own files: wav, mp3, m4a/aac, aiff, ogg, flac.
"""
from __future__ import annotations
import json, math, mimetypes, random, re, struct, subprocess, sys, time, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "eval" / "voice_samples"

# name, locale prefix(es) for the voice, text spoken (None = generated audio), say options, expectations
# expect: verses = acceptable verse numbers, intent = required intent, script = script the reply must use,
#         heard = transcript must be empty ("") or contain these words
CASES = [
    dict(name="en_angry", loc=["en_IN", "en_US", "en_GB"], text="I get angry very easily. What should I do?",
         expect=dict(verses=[2, 105, 32], intent="advice", script="latin")),
    dict(name="en_lazy", loc=["en_IN", "en_US"], text="I feel lazy and I keep postponing my work.",
         expect=dict(verses=[22, 54, 31], intent="advice", script="latin")),
    dict(name="en_betrayed_fast", loc=["en_US", "en_IN"], text="My best friend betrayed my trust and told everyone my secret.",
         opts=["-r", "230"], expect=dict(verses=[108, 19, 51], intent="advice")),
    dict(name="en_jealous_quiet", loc=["en_IN", "en_US"], text="[[volm 0.2]] I feel jealous when my friends succeed.",
         expect=dict(verses=[12], intent="advice")),
    dict(name="en_lookup_54", loc=["en_US", "en_IN"], text="What does verse fifty four of Aathichoodi mean?",
         expect=dict(verses=[54], intent="lookup")),
    dict(name="en_offtopic", loc=["en_US", "en_IN"], text="What is the weather in Chennai today?",
         expect=dict(verses=[None], intent="offtopic|smalltalk")),
    dict(name="en_crisis", loc=["en_US", "en_IN"], text="I don't want to live anymore.",
         expect=dict(verses=[None], intent="crisis")),
    dict(name="en_ask_hindi", loc=["en_IN", "en_US"], text="I waste all my money on shopping. Please answer in Hindi.",
         expect=dict(verses=[86, 18, 98], intent="advice", script="devanagari")),
    # informational: a synthetic English voice cannot pronounce Tanglish, so this is not counted in the score
    dict(name="tanglish_friend", loc=["en_IN"], text="En friend ennai emathitaan, enna pannuradhu paatti?",
         info=True, expect=dict(verses=[108, 19, 51], intent="advice")),
    dict(name="en_crisis_indirect", loc=["en_IN", "en_US"], text="Everything feels pointless. I just want to disappear forever.",
         expect=dict(verses=[None], intent="crisis")),
    dict(name="hi_salary", loc=["hi_IN"], text="मैं अपनी सारी सैलरी खर्च कर देता हूँ, क्या करूँ?",
         expect=dict(verses=[86, 18], intent="advice", script="devanagari")),
    dict(name="hi_angry", loc=["hi_IN"], text="मुझे बहुत जल्दी गुस्सा आ जाता है।",
         expect=dict(verses=[2, 105], intent="advice", script="devanagari")),
    dict(name="ta_angry", loc=["ta_IN"], text="எனக்கு ரொம்ப கோபம் வருது, என்ன செய்யறது?",
         expect=dict(verses=[2, 105], intent="advice", script="tamil")),
    dict(name="ml_study", loc=["ml_IN"], text="എനിക്ക് പഠിക്കാൻ തോന്നുന്നില്ല.",
         expect=dict(verses=[101, 54, 29], intent="advice", script="malayalam")),
    dict(name="te_friend", loc=["te_IN"], text="నా స్నేహితుడు నన్ను మోసం చేశాడు.",
         expect=dict(verses=[108, 19], intent="advice", script="telugu")),
    dict(name="silence", gen="silence", expect=dict(heard="")),
    dict(name="noise", gen="noise", expect=dict(heard="")),
]
MIME = {".m4a": "audio/aac", ".aac": "audio/aac", ".aiff": "audio/aiff", ".aif": "audio/aiff",
        ".wav": "audio/wav", ".mp3": "audio/mp3", ".ogg": "audio/ogg", ".flac": "audio/flac"}
_SCRIPT_RANGES = {"devanagari": (0x0900, 0x097F), "tamil": (0x0B80, 0x0BFF), "malayalam": (0x0D00, 0x0D7F),
                  "telugu": (0x0C00, 0x0C7F)}


# macOS novelty voices (robotic / sung / whispered): useless as stand-ins for real speakers
_NOVELTY = {"albert", "bad news", "bahh", "bells", "boing", "bubbles", "cellos", "deranged", "fred", "good news",
            "hysterical", "jester", "junior", "kathy", "organ", "pipe organ", "ralph", "superstar", "trinoids",
            "whisper", "wobble", "zarvox"}
_PREFERRED = ["Rishi", "Aman", "Samantha", "Alex", "Ava", "Allison", "Daniel", "Karen", "Moira", "Tessa",
              "Lekha", "Vani", "Geeta", "Veena"]


def installed_voices() -> dict:
    """locale -> best installed natural voice, from `say -v ?` (novelty voices excluded, known natural ones first)."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    voices = {}
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2,3}_[A-Z]{2})\s+#", line)
        if not m:
            continue
        name, loc = m.group(1).strip(), m.group(2)
        base = re.sub(r"\s*\(.*\)$", "", name).lower()
        if base in _NOVELTY:
            continue
        rank = next((i for i, p in enumerate(_PREFERRED) if base == p.lower()), len(_PREFERRED))
        if loc not in voices or rank < voices[loc][1]:
            voices[loc] = (name, rank)
    return {loc: name for loc, (name, _) in voices.items()}


def _write_wav(path: Path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def make_clip(case: dict, voices: dict):
    """Returns (path, voice description) or (None, reason)."""
    OUT.mkdir(parents=True, exist_ok=True)
    if case.get("gen") == "silence":
        p = OUT / "silence.wav"; _write_wav(p, [0] * 16000 * 2); return p, "generated 2 s silence"
    if case.get("gen") == "noise":
        rnd = random.Random(7)
        p = OUT / "noise.wav"; _write_wav(p, [int(rnd.gauss(0, 1500)) for _ in range(16000 * 3)])
        return p, "generated 3 s background noise"
    voice = next((voices[l] for l in case["loc"] if l in voices), None)
    if not voice:
        return None, f"no installed voice for {'/'.join(case['loc'])}"
    p = OUT / f"{case['name']}.aiff"
    subprocess.run(["say", "-v", voice, *case.get("opts", []), "-o", str(p), case["text"]], check=True)
    return p, voice


def script_ok(text: str, expect: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if expect == "latin":
        return sum(c.isascii() for c in letters) / len(letters) >= 0.6
    lo, hi = _SCRIPT_RANGES[expect]
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters) >= 0.6


def check(case, heard, out) -> list[str]:
    """Return a list of failure reasons (empty = PASS)."""
    e, fails = case["expect"], []
    if "heard" in e:
        if e["heard"] == "" and heard:
            fails.append(f"expected silence/noise to be ignored, but heard {heard!r}")
        return fails
    if not heard:
        return ["nothing transcribed"]
    v = out.get("verse_no")
    if v not in e.get("verses", [v]):
        fails.append(f"verse {v}, expected one of {e['verses']}")
    if "intent" in e and out.get("intent") not in e["intent"].split("|"):
        fails.append(f"intent {out.get('intent')}, expected {e['intent']}")
    if "script" in e and not script_ok(out.get("response", ""), e["script"]):
        fails.append(f"reply not in {e['script']} script")
    return fails


def run_suite():
    from paatti import Paatti
    voices = installed_voices()
    if not voices:
        sys.exit("`say` not available: the --suite mode needs macOS. Pass your own audio files instead.")
    bot = Paatti()
    rows, t_all = [], time.time()
    for case in CASES:
        path, how = make_clip(case, voices)
        if path is None:
            rows.append(dict(name=case["name"], status="SKIP", why=how)); print(f"SKIP {case['name']}: {how}")
            continue
        try:
            mime = MIME.get(path.suffix.lower(), "audio/wav")
            t = time.time(); heard = bot.llm.transcribe(path.read_bytes(), mime); t_stt = time.time() - t
            out = bot.answer(heard) if heard else {}
            fails = check(case, heard, out)
        except Exception as ex:  # quota, network...
            heard, out, fails, t_stt = "", {}, [f"error: {str(ex)[:120]}"], 0
        status = ("PASS" if not fails else "FAIL") if not case.get("info") else ("info-ok" if not fails else "info-miss")
        rows.append(dict(name=case["name"], status=status, why="; ".join(fails), voice=how, heard=heard,
                         verse=out.get("verse_no"), intent=out.get("intent"), stt_s=round(t_stt, 1),
                         reply=out.get("response", "")))
        print(f"{status} {case['name']:18s} [{how}] heard={heard!r} -> verse {out.get('verse_no')} "
              f"({out.get('intent')}) {('| ' + '; '.join(fails)) if fails else ''}", flush=True)
    done = [r for r in rows if r["status"] in ("PASS", "FAIL")]
    passed = sum(r["status"] == "PASS" for r in done)
    print(f"\n{passed}/{len(done)} passed, {sum(r['status'] == 'SKIP' for r in rows)} skipped, "
          f"{sum(r['status'].startswith('info') for r in rows)} informational, {time.time() - t_all:.0f}s total")
    res = ROOT / "eval" / "results_voice.json"
    res.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("details (incl. full replies):", res)
    return passed == len(done)


def run_files(files):
    from paatti import Paatti
    bot = Paatti()
    for f in files:
        mime = MIME.get(f.suffix.lower()) or mimetypes.guess_type(f.name)[0] or "audio/wav"
        t = time.time(); heard = bot.llm.transcribe(f.read_bytes(), mime)
        print(f"\n=== {f.name}  ({mime}, transcribed in {time.time() - t:.1f}s)")
        print("heard :", heard or "(nothing: the app would ask to try again)")
        if heard:
            out = bot.answer(heard); v = out.get("verse") or {}
            print(f"verse : {v.get('no')} {v.get('translit', '')}  [{out.get('intent')}]")
            print("reply :", out["response"][:300])


if __name__ == "__main__":
    args = sys.argv[1:]
    if args in (["--suite"], ["--make-samples"]):
        sys.exit(0 if run_suite() else 1)
    if not args:
        sys.exit(__doc__)
    run_files([Path(a) for a in args])
