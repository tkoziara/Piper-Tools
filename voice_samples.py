#!/usr/bin/env python3
"""Synthesize, list, and play piper TTS voice samples for given languages."""
import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import msvcrt

    def _getch():
        return msvcrt.getch().decode()
except ImportError:

    def _getch():
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch

from synth import (
    ROOT,
    VOICES_DIR,
    ensure_voice_available,
    infer_language_from_voice,
    local_voice_exists,
    safe_load_voice_index,
    synth,
)

DEFAULT_PHRASE_EN = (
    'Having bowed to Narayana and to the supreme man Nara, '
    'and to the goddess Sarasvati, then should the word "Jaya" be uttered.'
)
DEFAULT_PHRASE_PL = (
    'Pokłoniwszy się Narajanie i Najwyższemu Człowiekowi Narze, '
    'oraz bogini Saraswati, należy wypowiedzieć słowo „Dżaja”.'
)


def filter_voices_by_lang(voice_index, langs):
    langs_set = set(langs)
    matching = {}
    for key, meta in voice_index.items():
        lang_code = meta.get("language", {}).get("code", "")
        if lang_code:
            lang_prefix = lang_code.split("_")[0]
        else:
            lang_prefix = key.split("_")[0].split("-")[0]
        if lang_prefix in langs_set:
            matching[key] = meta
    return matching


def play_file(path: Path, prefix=""):
    p = str(path)
    for prog in ("aplay", "paplay", "ffplay.exe", "ffplay"):
        if shutil.which(prog):
            if "ffplay" in prog:
                cmd = [prog, "-autoexit", "-nodisp", p]
                kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
            else:
                cmd = [prog, p]
                kwargs = {}
            stop_animation = threading.Event()

            def animate():
                while not stop_animation.is_set():
                    for dots in (".", "..", "..."):
                        if stop_animation.is_set():
                            return
                        print(f"\r{prefix} {dots}", end="", flush=True)
                        time.sleep(0.3)

            t = threading.Thread(target=animate, daemon=True)
            t.start()

            proc = subprocess.Popen(cmd, **kwargs)
            skipped = False
            already_waited = False
            try:
                import tty, termios, select, os
                if sys.stdin.isatty():
                    fd = sys.stdin.fileno()
                    old = termios.tcgetattr(fd)
                    tty.setraw(fd)
                    try:
                        while proc.poll() is None:
                            r, _, _ = select.select([fd], [], [], 0.1)
                            if r and os.read(fd, 1) == b"n":
                                skipped = True
                                break
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
                else:
                    proc.wait()
                    already_waited = True
            except ImportError:
                import msvcrt
                while proc.poll() is None:
                    if msvcrt.kbhit() and msvcrt.getch().decode() == "n":
                        skipped = True
                        break

            if skipped:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            elif not already_waited:
                proc.wait()
            stop_animation.set()
            t.join()
            status = " \u2192 skipped" if skipped else ""
            print(f"\r{prefix}\033[K{status}")
            return not skipped
    raise SystemExit("No audio player found; install `aplay`, `paplay` or `ffplay`.")


def get_voice_samples(output_dir):
    wavs = sorted(output_dir.glob("*.wav"))
    samples = []
    for w in wavs:
        voice_name = w.stem
        samples.append((voice_name, w))
    return samples


def list_voices(output_dir):
    samples = get_voice_samples(output_dir)
    if not samples:
        print("No synthesized voice samples found in", output_dir)
        return
    print("Synthesized voice samples:")
    for idx, (voice_name, path) in enumerate(samples, start=1):
        size = path.stat().st_size
        print(f"  {idx:3d}. {voice_name} ({size} bytes)")


def load_approved(output_dir):
    p = output_dir / "approved.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def play_samples(output_dir, number=None, approve=False):
    samples = get_voice_samples(output_dir)
    if not samples:
        print("No synthesized voice samples found in", output_dir)
        return

    previously_approved = set(load_approved(output_dir))
    approved = list(previously_approved)

    indices = list(range(len(samples))) if number is None else [number - 1]
    i = 0
    try:
        while i < len(indices):
            idx = indices[i]
            if idx < 0 or idx >= len(samples):
                print(f"Invalid sample number {idx + 1}; valid range 1-{len(samples)}", file=sys.stderr)
                i += 1
                continue
            voice_name, path = samples[idx]

            status = ""
            if approve and previously_approved:
                status = " [previously approved]" if voice_name in previously_approved else " [previously skipped]"

            played = play_file(path, prefix=f"Playing #{idx + 1}: {voice_name}{status}")

            if not played:
                i += 1
                continue

            if not approve:
                i += 1
                continue

            print(f"Approve \"{voice_name}\"? [y -- approve / SPACE -- replay / g -- go to / other -- skip] ", end="", flush=True)
            ch = _getch()
            print()
            if ch == "y":
                approved.append(voice_name)
                print(f"  Approved: {voice_name}")
                i += 1
            elif ch == " ":
                continue
            elif ch == "g":
                min_n = indices[0] + 1
                max_n = indices[-1] + 1
                try:
                    target = input(f"  Go to sample ({min_n}-{max_n}): ").strip()
                    target_n = int(target)
                except (ValueError, EOFError):
                    print("  Invalid input")
                    continue
                if target_n < min_n or target_n > max_n:
                    print(f"  Out of range (valid: {min_n}-{max_n})")
                    continue
                target_idx = target_n - 1
                if target_idx in indices:
                    i = indices.index(target_idx)
                else:
                    print(f"  Sample #{target_n} is not in the current play list")
            else:
                print(f"  Skipped: {voice_name}")
                i += 1
    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    if approve:
        approved_path = output_dir / "approved.json"
        new_approved = [v for v in approved if v not in previously_approved]
        if new_approved or previously_approved:
            approved_path.write_text(json.dumps(approved, indent=2), encoding="utf-8")
            print(f"Approved voices written to {approved_path}")
        else:
            print("No approvals to save.")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize a phrase using all available piper voices for given languages.",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="en,pl",
        help="Comma-separated list of language codes (default: en,pl)",
    )
    parser.add_argument("--phrase", type=str, default=None,
                        help=f'Text to synthesize (default: "{DEFAULT_PHRASE_EN}" '
                             f'or "{DEFAULT_PHRASE_PL}" for Polish voices)')
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./voice_samples"),
        help="Output directory for WAV files (default: ./voice_samples)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List synthesized voice samples that exist in the output directory",
    )
    parser.add_argument(
        "--play",
        nargs="?",
        const=-1,
        type=int,
        metavar="N",
        help="Play a synthesized voice sample by number (from --list); "
             "omit number to play all samples",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Prompt to approve each voice after playback (use with --play)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()

    if args.approve and args.play is None:
        print("Error: --approve requires --play", file=sys.stderr)
        sys.exit(1)

    if args.list or args.play is not None:
        if args.list:
            list_voices(output_dir)
        if args.play is not None:
            if not output_dir.exists():
                print(f"Output directory '{output_dir}' does not exist.", file=sys.stderr)
                sys.exit(1)
            n = None if args.play == -1 else args.play
            play_samples(output_dir, n, approve=args.approve)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    print("Loading voice index...")
    voice_index = safe_load_voice_index()
    if not voice_index:
        print("Warning: remote voice index unavailable.", file=sys.stderr)
        sys.exit(1)

    matching_voices = filter_voices_by_lang(voice_index, langs)
    if not matching_voices:
        print(f"No voices found for languages: {langs}")
        sys.exit(1)

    print(f"Found {len(matching_voices)} voice models for {langs}")
    for key in sorted(matching_voices):
        print(f"  {key}")

    success = 0
    fail = 0
    for voice_name in sorted(matching_voices):
        phrase = args.phrase
        if phrase is None:
            lang = infer_language_from_voice(voice_name, voice_index)
            phrase = DEFAULT_PHRASE_PL if lang == "pl" else DEFAULT_PHRASE_EN

        outpath = output_dir / f"{voice_name}.wav"
        if outpath.exists():
            print(f"Skipping {voice_name} -> {outpath.name} (already exists)")
            success += 1
            continue

        print(f"\n--- {voice_name} ---")
        if not local_voice_exists(voice_name, VOICES_DIR):
            try:
                ensure_voice_available(voice_name, VOICES_DIR)
            except SystemExit as e:
                print(f"  Failed to download voice: {e}", file=sys.stderr)
                fail += 1
                continue

        print(f"  Synthesizing -> {outpath.name}")
        try:
            synth(voice_name, phrase, outpath, VOICES_DIR)
            success += 1
        except Exception as e:
            print(f"  Synthesis failed: {e}", file=sys.stderr)
            fail += 1

    print(f"\nDone. {success} succeeded, {fail} failed. Output in: {output_dir}")


if __name__ == "__main__":
    main()
