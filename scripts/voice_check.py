"""Test speech -> transcript -> Paatti reply with your real Gemini key, without the browser.

  python3 scripts/voice_check.py my_clip.m4a other_clip.wav
  python3 scripts/voice_check.py --make-samples      # macOS: creates test clips with the built-in `say` voices

Supported formats: wav, mp3, m4a/aac, aiff, ogg, flac.
"""
import mimetypes, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES = [  # (file, macOS voice, text)
    ("en_angry.aiff", "Rishi", "I get angry very easily. What should I do?"),
    ("en_lazy.aiff", "Samantha", "I feel lazy and I keep postponing my work."),
    ("hi_salary.aiff", "Lekha", "मैं अपनी सारी सैलरी खर्च कर देता हूँ, क्या करूँ?"),
    ("silence.aiff", None, None),
]
MIME = {".m4a": "audio/aac", ".aac": "audio/aac", ".aiff": "audio/aiff", ".aif": "audio/aiff",
        ".wav": "audio/wav", ".mp3": "audio/mp3", ".ogg": "audio/ogg", ".flac": "audio/flac"}


def make_samples(out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    made = []
    for name, voice, text in SAMPLES:
        f = out_dir / name
        try:
            if text is None:  # 2 s of silence
                subprocess.run(["say", "-o", str(f), "[[slnc 2000]]"], check=True)
            else:
                subprocess.run(["say", "-v", voice, "-o", str(f), text], check=True)
            made.append(f)
            print("made", f.name)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"skip {name}: {e} (voice '{voice}' may not be installed: System Settings > Accessibility > Spoken Content)")
    return made


def main():
    args = sys.argv[1:]
    files = make_samples(ROOT / "eval" / "voice_samples") if args == ["--make-samples"] else [Path(a) for a in args]
    if not files:
        sys.exit(__doc__)
    from paatti import Paatti
    bot = Paatti()
    for f in files:
        mime = MIME.get(f.suffix.lower()) or mimetypes.guess_type(f.name)[0] or "audio/wav"
        t = time.time()
        heard = bot.llm.transcribe(f.read_bytes(), mime).strip()
        print(f"\n=== {f.name}  ({mime}, transcribed in {time.time() - t:.1f}s)")
        print("heard :", heard or "(nothing: the app would ask to try again)")
        if heard:
            out = bot.answer(heard)
            v = out.get("verse") or {}
            print(f"verse : {v.get('no')} {v.get('translit', '')}  [{out.get('intent')}]")
            print("reply :", out["response"][:300])


if __name__ == "__main__":
    main()
