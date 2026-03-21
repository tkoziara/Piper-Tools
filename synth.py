#!/usr/bin/env python3
"""Simple runtime script to synthesize two sample texts using the piper CLI.

This script assumes the `piper` package is available on `python3 -m piper` and
that voice model folders/files are present in the `voices/` directory.
"""
import subprocess
import sys
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VOICES_DIR = ROOT / "voices"


def play_file(path: Path):
    """Play a WAV file with whatever player is available on the system.

    Tries `aplay`, `paplay`, then `ffplay` (from ffmpeg) and exits with an
    error if none can be found. This is deliberately small and avoids adding
    any extra Python dependencies.
    """
    p = str(path)
    for prog in ("aplay", "paplay", "ffplay"):
        if shutil.which(prog):
            if prog == "ffplay":
                cmd = [prog, "-autoexit", "-nodisp", p]
            else:
                cmd = [prog, p]
            print("Playing audio with", prog)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                pass
            return
    raise SystemExit("No audio player found; install `aplay`, `paplay` or `ffplay`.")

VOICES = {
    "en": "en_US-lessac-medium",
    "pl": "pl_PL-gosia-medium",
}

def synth(voice_name: str, text: str, outpath: Path):
    cmd = [
        sys.executable,
        "-m",
        "piper",
        "-m",
        voice_name,
        "--data-dir",
        str(VOICES_DIR),
        "-f",
        str(outpath),
        "--",
        text,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(
        description="Simple wrapper around `piper` for synthesizing text."
    )
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to an ONNX model file. If provided the "
                             "script will invoke train.py synth-test on that "
                             "file (preferred for intermediate checkpoints).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--voice", "-m", type=str, default=None,
                        help="Voice/model name (e.g. en_US-lessac-medium). "
                             "Ignored if --model is given.")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Text string to synthesize. If omitted the script "
                             "will read sample_en.txt or sample_pl.txt based on "
                             "language deduced from voice name or model path.")
    parser.add_argument("--out-file", "-f", type=Path, default=None,
                        help="Output WAV file path. Defaults to out_<lang>.wav")
    parser.add_argument("--voices-dir", type=Path, default=VOICES_DIR,
                        help="Directory containing voice models.")
    parser.add_argument("--list", action="store_true",
                        help="List available sample voices and exit.")
    parser.add_argument("--play", action="store_true",
                        help="Play the output WAV instead of (or in addition to) writing a file."
                             "Requires `aplay`/`paplay`/`ffplay` on PATH.")
    args = parser.parse_args()

    # list available voices
    if args.list:
        print("Available sample voices:")
        for lang, name in VOICES.items():
            print(f"  {lang}: {name}")
        sys.exit(0)

    # if a concrete onnx model was provided, delegate to train.py synth-test
    if args.model is not None:
        model_path = args.model.resolve()
        if not model_path.exists():
            raise SystemExit(f"Model file not found: {model_path}")
        # decide output
        if args.out_file is not None:
            outpath = args.out_file
        else:
            outpath = ROOT / (model_path.stem + ".wav")
        # build command
        cmd = [
            sys.executable,
            "train.py",
            "synth-test",
            "--model",
            str(model_path),
            "--text",
            args.text or "",
            "--out-file",
            str(outpath),
        ]
        print("Running synth-test via train.py:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("Done.")
        if args.play:
            play_file(outpath)
        return

    # determine voice
    voice = args.voice
    if voice is None:
        voice = VOICES.get("en")
        lang = "en"
    else:
        # try to infer language key from voices dict
        lang = None
        for k, v in VOICES.items():
            if v == voice:
                lang = k
                break
        if lang is None:
            # guess from prefix
            lang = "en" if voice.startswith("en") else "pl"

    # determine text
    if args.text is not None:
        text = args.text
    else:
        sample_file = ROOT / ("sample_" + lang + ".txt")
        text = sample_file.read_text(encoding="utf-8").strip()

    # determine output file
    if args.out_file is not None:
        outpath = args.out_file
    else:
        outpath = ROOT / f"out_{lang}.wav"

    print(f"Synthesizing {lang} using voice '{voice}' to {outpath}...")
    synth(voice, text, outpath)
    print("Done.")
    if args.play:
        play_file(outpath)

if __name__ == "__main__":
    main()
