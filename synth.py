#!/usr/bin/env python3
"""Simple runtime script to synthesize two sample texts using the piper CLI.

This script assumes the `piper` package is available on `python3 -m piper` and
that voice model folders/files are present in the `voices/` directory.
"""
import json
import re
import subprocess
import sys
import argparse
import shutil
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VOICES_DIR = ROOT / "voices"
VOICE_INDEX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
VOICE_DOWNLOAD_URL_FORMAT = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "{lang_family}/{lang_code}/{voice_name}/{voice_quality}/"
    "{lang_code}-{voice_name}-{voice_quality}{extension}?download=true"
)
VOICE_DOWNLOAD_PATTERN = re.compile(
    r"^(?P<lang_family>[^-]+)_(?P<lang_region>[^-]+)-(?P<voice_name>[^-]+)-(?P<voice_quality>.+)$"
)


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


def load_voice_index():
    request = urllib.request.Request(
        VOICE_INDEX_URL,
        headers={"User-Agent": "Piper-Tools/1.0"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def safe_load_voice_index():
    try:
        return load_voice_index()
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        print(
            f"Warning: failed to load remote voice index from {VOICE_INDEX_URL}: {exc}",
            file=sys.stderr,
        )
        return {}


def build_alias_map(voice_index):
    aliases = {}
    for key, meta in voice_index.items():
        aliases[key] = key
        for alias in meta.get("aliases", []):
            aliases[alias] = key
    return aliases


def resolve_voice_name(voice, voice_index, alias_map):
    if voice in voice_index:
        return voice
    if voice in alias_map:
        return alias_map[voice]
    voice_lower = voice.lower()
    for key in voice_index:
        if key.lower() == voice_lower:
            return key
    for alias, target in alias_map.items():
        if alias.lower() == voice_lower:
            return target
    return voice


def resolve_voice_selection(voice, voice_index, alias_map):
    if voice_index and voice.isdigit():
        index = int(voice)
        keys = sorted(voice_index)
        if 1 <= index <= len(keys):
            print(f"Selected voice #{index}: {keys[index - 1]}")
            return keys[index - 1]
        raise SystemExit(
            f"Voice index {voice} is out of range (1-{len(keys)}). Run --list to see available voices."
        )
    return resolve_voice_name(voice, voice_index, alias_map)


def infer_language_from_voice(voice, voice_index=None):
    if voice_index is not None and voice in voice_index:
        lang_code = voice_index[voice].get("language", {}).get("code", "")
        if lang_code:
            return lang_code.split("_")[0]
    match = re.match(r"^([a-z]{2})(?:_[A-Z]{2})?-", voice)
    if match:
        return match.group(1)
    return "en"


def download_file_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        total = int(response.getheader("Content-Length") or 0)
        with open(dest, "wb") as out_file:
            downloaded = 0
            chunk_size = 8192
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total
                    bar_len = 40
                    filled = int(pct * bar_len)
                    bar = "#" * filled + "-" * (bar_len - filled)
                    print(
                        f"\r  Downloading {dest.name}: [{bar}] {downloaded}/{total} bytes",
                        end="",
                        flush=True,
                    )
            if total:
                print()
            else:
                print(f"  Downloaded {downloaded} bytes to {dest}")


def download_voice(voice: str, download_dir: Path) -> None:
    match = VOICE_DOWNLOAD_PATTERN.match(voice)
    if not match:
        raise ValueError(
            f"Voice '{voice}' did not match the expected pattern: "
            "<lang_family>_<lang_region>-<voice_name>-<voice_quality>"
        )

    lang_family = match.group("lang_family")
    lang_region = match.group("lang_region")
    lang_code = f"{lang_family}_{lang_region}"
    voice_name = match.group("voice_name")
    voice_quality = match.group("voice_quality")
    for extension in (".onnx", ".onnx.json"):
        url = VOICE_DOWNLOAD_URL_FORMAT.format(
            lang_family=lang_family,
            lang_code=lang_code,
            voice_name=voice_name,
            voice_quality=voice_quality,
            extension=extension,
        )
        dest = download_dir / f"{lang_code}-{voice_name}-{voice_quality}{extension}"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  Skipping existing file: {dest.name}")
            continue
        print(f"Downloading voice file: {dest.name}")
        download_file_with_progress(url, dest)


def local_voice_exists(voice: str, voices_dir: Path) -> bool:
    return (
        (voices_dir / f"{voice}.onnx").exists()
        and (voices_dir / f"{voice}.onnx.json").exists()
    )


def ensure_voice_available(voice: str, voices_dir: Path) -> None:
    if local_voice_exists(voice, voices_dir):
        return
    print(f"Voice '{voice}' is missing locally. Downloading to '{voices_dir}'...")
    try:
        download_voice(voice, voices_dir)
    except Exception as exc:
        raise SystemExit(
            f"Failed to download voice '{voice}': {exc}\n"
            "Please check your network connection and try again."
        )


def list_voices(voice_index):
    if voice_index:
        print("Available voices from rhasspy/piper-voices:")
        print("Use '--voice <number>' or '--voice <voice-name>' to select a voice.")
        for idx, key in enumerate(sorted(voice_index), start=1):
            meta = voice_index[key]
            lang = meta.get("language", {}).get("code", "")
            quality = meta.get("quality", "")
            name = meta.get("name", "")
            details = []
            if lang:
                details.append(lang)
            if name:
                details.append(name)
            if quality:
                details.append(quality)
            if details:
                print(f"  {idx}. {key} ({', '.join(details)})")
            else:
                print(f"  {idx}. {key}")
    else:
        print("Available sample voices:")
        for lang, name in VOICES.items():
            print(f"  {lang}: {name}")


def synth(voice_name: str, text: str, outpath: Path, voices_dir: Path):
    cmd = [
        sys.executable,
        "-m",
        "piper",
        "-m",
        voice_name,
        "--data-dir",
        str(voices_dir),
        "-f",
        str(outpath),
        "--",
        text,
    ]
    print("Running:", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr or ""
        stdout = completed.stdout or ""
        if "Unable to find voice" in stderr:
            message = (
                f"Voice '{voice_name}' was not found in the local voices directory '{voices_dir}'.\n"
                "Please install that voice model before running synth.py, for example:\n"
                f"  .venv/bin/python -m piper download_voices {voice_name}\n"
                "or place the voice model files under the voices directory."
            )
            if stdout.strip():
                message += f"\n\nAdditional output:\n{stdout.strip()}"
            raise SystemExit(message)
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )

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
                        help="Voice/model name or list number (e.g. 5 or en_US-lessac-medium). "
                             "The script will also resolve known aliases from the "
                             "remote piper-voices index. Ignored if --model is given.")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Text string to synthesize. If omitted the script "
                             "will read sample_en.txt or sample_pl.txt based on "
                             "language deduced from the voice name.")
    parser.add_argument("--out-file", "-f", type=Path, default=None,
                        help="Output WAV file path. Defaults to out_<lang>.wav")
    parser.add_argument("--voices-dir", type=Path, default=VOICES_DIR,
                        help="Directory containing voice models.")
    parser.add_argument("--list", action="store_true",
                        help="List available voices from rhasspy/piper-voices and exit.")
    parser.add_argument("--play", action="store_true",
                        help="Play the output WAV instead of (or in addition to) writing a file."
                             "Requires `aplay`/`paplay`/`ffplay` on PATH.")
    args = parser.parse_args()

    voice_index = None
    alias_map = None
    if args.list or args.voice is not None:
        voice_index = safe_load_voice_index()
        alias_map = build_alias_map(voice_index)

    if args.list:
        list_voices(voice_index)
        sys.exit(0)

    # allow numeric selection from --list output
    if args.voice is not None and voice_index is not None:
        args.voice = resolve_voice_selection(args.voice, voice_index, alias_map)

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
        if voice_index is not None:
            resolved = resolve_voice_name(voice, voice_index, alias_map)
            if resolved != voice:
                print(f"Resolved voice alias {voice!r} to canonical model {resolved!r}")
            voice = resolved
        lang = infer_language_from_voice(voice, voice_index)
        ensure_voice_available(voice, args.voices_dir)

    # determine text
    if args.text is not None:
        text = args.text
    else:
        sample_file = ROOT / ("sample_" + lang + ".txt")
        if not sample_file.exists():
            raise SystemExit(
                f"No sample text available for language '{lang}'. Use --text to provide a sentence."
            )
        text = sample_file.read_text(encoding="utf-8").strip()

    # determine output file
    if args.out_file is not None:
        outpath = args.out_file
    else:
        outpath = ROOT / f"out_{lang}.wav"

    print(f"Synthesizing {lang} using voice '{voice}' to {outpath}...")
    synth(voice, text, outpath, args.voices_dir)
    print("Done.")
    if args.play:
        play_file(outpath)

if __name__ == "__main__":
    main()
