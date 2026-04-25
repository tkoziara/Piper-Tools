#!/usr/bin/env python3
"""Generate approved sentence-level samples from audio files using Whisper."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tty
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_PADDING = 0.20
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3"}
SENTENCE_REGEX = re.compile(r"(.+?[.!?])(?:\s+|$)", re.DOTALL)
SESSION_FILENAME = "session.json"
CANDIDATE_PREFIX = "candidate_"


def find_audio_paths(inputs: List[str], recursive: bool = False) -> List[Path]:
    paths: List[Path] = []
    if not inputs:
        inputs = ["."]
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            if recursive:
                candidates = [x for x in p.rglob("*") if x.suffix.lower() in SUPPORTED_AUDIO_EXTS]
            else:
                candidates = [x for x in p.iterdir() if x.suffix.lower() in SUPPORTED_AUDIO_EXTS]
            paths.extend(sorted(candidates))
        elif p.is_file():
            if p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                paths.append(p)
            else:
                raise SystemExit(f"Unsupported file extension: {p}")
        else:
            raise SystemExit(f"Input path not found: {p}")
    if not paths:
        raise SystemExit("No audio files found in the provided inputs.")
    return paths


def detect_default_model() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "medium"
    except Exception:
        pass
    return "small"


def load_whisper_model(model_name: str):
    try:
        import whisper
    except ImportError as exc:
        raise SystemExit(
            "Whisper is not installed. Install it with `pip install -U openai-whisper`."
        ) from exc
    try:
        return whisper.load_model(model_name)
    except Exception as exc:
        raise SystemExit(f"Failed to load Whisper model '{model_name}': {exc}") from exc


def parse_padding(value: str) -> float:
    padding = float(value)
    if padding < 0.1:
        raise argparse.ArgumentTypeError("padding must be at least 0.1 seconds")
    return padding


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def input_with_prefill(prompt: str, text: str) -> str:
    try:
        import readline
    except ImportError:
        return input(prompt)

    def hook() -> None:
        readline.insert_text(text)

    readline.set_startup_hook(hook)
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook(None)


class Spinner:
    def __init__(self, message: str, delay: float = 0.1):
        self.message = message
        self.delay = delay
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        symbols = "|/-\\"
        idx = 0
        while not self.stop_event.is_set():
            sys.__stdout__.write(f"\r{self.message} {symbols[idx % len(symbols)]}")
            sys.__stdout__.flush()
            idx += 1
            time.sleep(self.delay)
        sys.__stdout__.write("\r" + " " * (len(self.message) + 4) + "\r")
        sys.__stdout__.flush()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join()


def ensure_candidate_metadata(candidate: Dict[str, Any]) -> None:
    if "original_start" not in candidate:
        candidate["original_start"] = candidate["start"]
    if "original_end" not in candidate:
        candidate["original_end"] = candidate["end"]
    if "merge_history" not in candidate:
        candidate["merge_history"] = []


def init_denoiser() -> tuple:
    try:
        with suppress_output():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                from df.enhance import init_df
    except ImportError as exc:
        raise SystemExit(
            "Denoising requested, but DeepFilterNet is not installed. "
            "Install it with `pip install deepfilternet`."
        ) from exc
    with suppress_output():
        model, df_state, _ = init_df()
    return model, df_state


def denoise_audio(src: Path, dst: Path, denoiser: Optional[tuple]) -> None:
    if denoiser is None:
        shutil.copy2(src, dst)
        return
    with suppress_output():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            from df.enhance import enhance, load_audio, save_audio

        model, df_state = denoiser
        audio, _ = load_audio(str(src), sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        save_audio(str(dst), enhanced, df_state.sr())


def rebuild_candidate_wav(
    candidate: Dict[str, Any],
    sample_rate: int,
    padding: float,
    max_end: Optional[float] = None,
) -> None:
    source = Path(candidate["source"])
    wav_path = Path(candidate["wav_path"])
    normalize_audio_segment(
        source,
        wav_path,
        candidate["start"],
        candidate["end"],
        sample_rate,
        padding=padding,
        max_end=max_end,
    )


def save_candidate_text(candidate: Dict[str, Any]) -> None:
    Path(candidate["txt_path"]).write_text(candidate["text"], encoding="utf-8")


def merge_forward_text(first: str, second: str) -> str:
    first = first.rstrip()
    if first and first[-1] in ".!?":
        first = first[:-1].rstrip()
    second = second.lstrip()
    if second:
        second = second[0].lower() + second[1:]
    return f"{first} {second}"


def get_audio_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return float(proc.stdout.strip())


def detect_trailing_silence_start(
    path: Path,
    silence_threshold: str = "-40dB",
    silence_duration: float = 0.1,
) -> Optional[float]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-af",
        f"silencedetect=noise={silence_threshold}:d={silence_duration}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    duration = get_audio_duration(path)
    last_silence_start: Optional[float] = None
    last_silence_end: Optional[float] = None
    for line in proc.stderr.splitlines():
        silence_start = re.search(r"silence_start: (\d+(?:\.\d+)?)", line)
        if silence_start:
            last_silence_start = float(silence_start.group(1))
            last_silence_end = None
        silence_end = re.search(r"silence_end: (\d+(?:\.\d+)?).*", line)
        if silence_end:
            last_silence_end = float(silence_end.group(1))
    if last_silence_start is None:
        return None
    if last_silence_end is not None:
        if last_silence_end >= duration - 0.05:
            return last_silence_start
        return None
    if last_silence_start >= duration - 0.05:
        return last_silence_start
    return None


def normalize_audio_segment(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    sample_rate: int,
    padding: float = 0.0,
    max_end: Optional[float] = None,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    padded_start = max(0.0, start - padding)
    padded_end = max(padded_start, end + padding)
    if max_end is not None:
        padded_end = min(padded_end, max_end)

    if padded_end <= padded_start + 1e-3:
        padded_end = padded_start + 0.01

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ss",
            str(padded_start),
            "-to",
            str(padded_end),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(temp_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    final_end: Optional[float] = None
    trailing_silence_start = detect_trailing_silence_start(temp_path)
    if trailing_silence_start is not None:
        final_end = min(padded_end - padded_start, trailing_silence_start + padding)

    if final_end is None or final_end >= (padded_end - padded_start) - 0.01:
        shutil.move(temp_path, dst)
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_path),
                "-ss",
                "0",
                "-to",
                str(final_end),
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                str(dst),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        temp_path.unlink()


def split_text_into_sentences(text: str) -> Tuple[List[str], str]:
    pieces = SENTENCE_REGEX.findall(text.strip())
    remainder = ""
    if pieces:
        consumed = sum(len(item) for item in pieces)
        remainder = text.strip()[consumed:].strip()
        return [item.strip() for item in pieces if item.strip()], remainder
    return [], text.strip()


def timing_for_sentence(sentence: str, text: str, start: float, end: float, offset: int) -> Tuple[float, float]:
    if not text or start == end:
        return start, end
    total = len(text)
    if total == 0:
        return start, end
    sentence_len = len(sentence)
    ratio_start = offset / total
    ratio_end = (offset + sentence_len) / total
    sentence_start = start + (end - start) * ratio_start
    sentence_end = start + (end - start) * ratio_end
    return sentence_start, sentence_end


def build_sentence_candidates(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    pending_text = ""
    pending_start = 0.0
    pending_end = 0.0
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if pending_text:
            text = pending_text + " " + text
            start = pending_start
        sentences, remainder = split_text_into_sentences(text)
        chunk_text = text
        current_offset = 0
        for sentence in sentences:
            sentence_start, sentence_end = timing_for_sentence(
                sentence,
                chunk_text,
                start,
                end,
                current_offset,
            )
            candidates.append(
                {
                    "text": sentence,
                    "start": sentence_start,
                    "end": sentence_end,
                }
            )
            current_offset += len(sentence)
        if remainder:
            pending_text = remainder
            pending_start = start + (end - start) * (current_offset / max(len(chunk_text), 1))
            pending_end = end
        else:
            pending_text = ""
            pending_start = 0.0
            pending_end = 0.0
    if pending_text:
        candidates.append(
            {
                "text": pending_text.strip(),
                "start": pending_start,
                "end": pending_end if pending_end > pending_start else pending_start + 1.0,
            }
        )
    return candidates


def make_candidate_files(
    audio_path: Path,
    candidates: List[Dict[str, Any]],
    scratch_dir: Path,
    sample_rate: int,
    padding: float,
) -> List[Dict[str, Any]]:
    audio_base = audio_path.stem
    candidate_dir = scratch_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        base_name = f"{CANDIDATE_PREFIX}{idx}"
        wav_path = candidate_dir / f"{base_name}.wav"
        txt_path = candidate_dir / f"{base_name}.txt"
        normalize_audio_segment(
            audio_path,
            wav_path,
            candidate["start"],
            candidate["end"],
            sample_rate,
            padding=padding,
        )
        txt_path.write_text(candidate["text"], encoding="utf-8")
        result.append(
            {
                "id": idx,
                "text": candidate["text"],
                "wav_path": str(wav_path),
                "txt_path": str(txt_path),
                "decision": "pending",
            }
        )
    return result


def save_session(session_path: Path, data: Dict[str, Any]) -> None:
    session_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_session(session_path: Path) -> Dict[str, Any]:
    return json.loads(session_path.read_text(encoding="utf-8"))


def find_last_sample_index(samples_dir: Path) -> int:
    highest = 0
    if not samples_dir.exists():
        return 0
    for child in samples_dir.iterdir():
        if child.is_file():
            m = re.match(r"sample_(\d+)\.(wav|txt)$", child.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest


def play_audio(path: Path, candidate: Optional[Dict[str, Any]] = None, padding: float = 0.0, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    player = None
    for prog in ("aplay", "paplay", "ffplay"):
        if shutil.which(prog):
            player = prog
            break
    if not player:
        raise SystemExit("No audio player found. Install aplay, paplay, or ffplay.")

    def format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02}:{secs:02}"

    def print_status(current: float, total: float, show_trim_end: bool = False) -> None:
        bar_width = 30
        progress = min(max(current / max(total, 1e-6), 0.0), 1.0)
        filled = int(progress * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        trim_end_indicator = "| [=] trim end " if show_trim_end else "               "
        print(
            f"\r[{bar}] {format_time(current)} / {format_time(total)} {trim_end_indicator}",
            end="",
            flush=True,
        )

    if player != "ffplay":
        subprocess.run([player, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    duration = get_audio_duration(path)
    position = 0.0
    stop_event = threading.Event()
    process: Optional[subprocess.Popen] = None
    paused = False
    finished = False
    start_time = 0.0

    def start_playback(start_position: float) -> subprocess.Popen:
        return subprocess.Popen(
            [
                player,
                "-autoexit",
                "-nodisp",
                "-loglevel",
                "quiet",
                "-ss",
                str(start_position),
                str(path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def playback_monitor() -> None:
        nonlocal position, start_time, paused
        while not stop_event.is_set():
            if paused or process is None or process.poll() is not None:
                time.sleep(0.1)
                continue
            elapsed = time.monotonic() - start_time
            current = min(position + elapsed, duration)
            print_status(current, duration)
            time.sleep(0.1)

    monitor_thread = threading.Thread(target=playback_monitor, daemon=True)
    monitor_thread.start()

    try:
        process = start_playback(position)
        start_time = time.monotonic()
        print("Press [q] quit, [p or space] pause/resume, [←] rewind, [→] forward, [-] step back, [u] untrim")
        while True:
            if process is not None and process.poll() is not None:
                position = duration
                process = None
                paused = False
                if not finished:
                    finished = True
                    print_status(position, duration)
                    print("\nPlayback finished. Press [p] to restart, [q] quit, [←] rewind, [→] forward, [-] step back, [u] untrim")
            key = read_single_key()
            if key == "q":
                break
            elif key == "p" or key == " ":
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()
                    elapsed = time.monotonic() - start_time
                    position = min(position + elapsed, duration)
                    paused = True
                    process = None
                    print_status(position, duration, show_trim_end=True)
                else:
                    if position >= duration:
                        position = 0.0
                    process = start_playback(position)
                    start_time = time.monotonic()
                    paused = False
                    finished = False
            elif key == "=":
                if paused:
                    trim_target = min(duration, position + padding)
                    temp_path = Path(tempfile.mktemp(suffix=".wav"))
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            str(path),
                            "-ss",
                            "0",
                            "-to",
                            str(trim_target),
                            str(temp_path),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    shutil.move(str(temp_path), str(path))
                    if candidate is not None:
                        candidate["end"] = min(candidate.get("end", duration), position + padding)
                    duration = get_audio_duration(path)
                    print_status(position, duration)
                    print("  Trimmed", end="", flush=True)
                else:
                    print("\nPause playback before trimming.")
            elif key == "-":
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
                if candidate is not None:
                    candidate["start"] = max(0.0, candidate["start"] - padding)
                    rebuild_candidate_wav(candidate, sample_rate, padding)
                position = 0.0
                process = start_playback(position)
                start_time = time.monotonic()
                paused = False
                finished = False
                print("Press [q] quit, [p] pause/resume, [←] rewind, [→] forward, [-] step back, [u] untrim")
            elif key == "u":
                if candidate is None:
                    print("\nNo candidate to untrim.")
                else:
                    original_start = candidate.get("original_start", candidate["start"])
                    original_end = candidate.get("original_end", candidate["end"])
                    if candidate["start"] == original_start and candidate["end"] == original_end:
                        print("\nAlready untrimmed.")
                    else:
                        if process is not None and process.poll() is None:
                            process.kill()
                            process.wait()
                        candidate["start"] = original_start
                        candidate["end"] = original_end
                        rebuild_candidate_wav(candidate, sample_rate, padding)
                        duration = get_audio_duration(path)
                        position = 0.0
                        process = start_playback(position)
                        start_time = time.monotonic()
                        paused = False
                        finished = False
                        print("\nUntrimmed and restarted playback.")
            elif key == "\x1b":
                second = read_single_key()
                third = read_single_key()
                if second == "[" and third == "D":
                    if process is not None and process.poll() is None:
                        process.kill()
                        process.wait()
                    if paused:
                        position = max(0.0, position - 2.0)
                    else:
                        elapsed = time.monotonic() - start_time
                        position = min(position + elapsed, duration)
                        position = max(0.0, position - 2.0)
                    process = start_playback(position)
                    start_time = time.monotonic()
                    paused = False
                elif second == "[" and third == "C":
                    if process is not None and process.poll() is None:
                        process.kill()
                        process.wait()
                    if paused:
                        position = min(duration, position + 2.0)
                    else:
                        elapsed = time.monotonic() - start_time
                        position = min(position + elapsed, duration)
                        position = min(duration, position + 2.0)
                    if position >= duration:
                        position = duration
                        process = None
                        paused = False
                        finished = True
                        print_status(position, duration)
                    else:
                        process = start_playback(position)
                        start_time = time.monotonic()
                        paused = False
            if process is not None and process.poll() is not None:
                break
    finally:
        stop_event.set()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        monitor_thread.join()
        print()


def read_single_key() -> str:
    if os.name == "nt":
        try:
            import msvcrt

            key = msvcrt.getwch()
            return key
        except Exception:
            pass

    fd = sys.stdin.fileno()
    old_settings = tty.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        tty.tcsetattr(fd, tty.TCSADRAIN, old_settings)
    return key


def prompt_decision(candidate: Optional[Dict[str, Any]] = None) -> str:
    commands = [
        "[a] approve",
        "[r] reject",
        "[p] play",
        "[b] back",
        "[n] forward",
        "[e] edit text",
        "[-] step back",
        "[=] trim end",
        "[u] untrim",
        "[+] merge forward",
        "[x] export approved",
        "[esc] quit",
    ]
    if candidate and candidate.get("merge_history"):
        commands.append("[m] unmerge")
    print("Commands: " + ", ".join(commands))
    print("Choice: ", end="", flush=True)
    key = read_single_key()
    if key == "":
        return "a"
    if key == "\x1b":
        print()
        return key
    print(key)
    return key.strip().lower()[:1]


def review_candidates(
    candidates: List[Dict[str, Any]],
    sample_rate: int,
    padding: float,
    session: Dict[str, Any],
    scratch_dir: Path,
    samples_dir: Path,
    start_idx: int = 0,
) -> None:
    for candidate in candidates:
        ensure_candidate_metadata(candidate)
    session_path = Path(candidates[0]["wav_path"]).parent.parent / SESSION_FILENAME if candidates else Path(SESSION_FILENAME)
    idx = min(max(0, start_idx), len(candidates) - 1) if candidates else 0
    while idx < len(candidates):
        candidate = candidates[idx]
        total = len(candidates)
        print(f"\n[{idx + 1}/{total}] {candidate['text']}")
        print(f"Current decision: {candidate['decision']}")
        action = prompt_decision(candidate)
        if action == "a":
            candidate["decision"] = "approved"
        elif action == "r":
            candidate["decision"] = "rejected"
        elif action == "p":
            play_audio(Path(candidate["wav_path"]), candidate=candidate, padding=padding, sample_rate=sample_rate)
        elif action == "b":
            if idx > 0:
                idx -= 1
            else:
                print("Already at first sample.")
        elif action == "n":
            if idx + 1 < len(candidates):
                idx += 1
            else:
                print("Already at last sample.")
        elif action == "e":
            print(f"Current text: {candidate['text']}")
            edited = input_with_prefill("New text (leave blank to keep current): ", candidate["text"]).strip()
            if edited and edited != candidate["text"]:
                candidate["text"] = edited
                save_candidate_text(candidate)
        elif action == "x":
            export_approved_samples(candidates, samples_dir, session, scratch_dir)
        elif action == "-":
            candidate["start"] = max(0.0, candidate["start"] - padding)
            next_start = None
            if idx + 1 < len(candidates):
                next_candidate = candidates[idx + 1]
                if next_candidate["source"] == candidate["source"]:
                    next_start = next_candidate["start"] - padding
            rebuild_candidate_wav(candidate, sample_rate, padding, max_end=next_start)
        elif action == "=":
            candidate["end"] = max(candidate["start"] + 0.01, candidate["end"] - padding)
            next_start = None
            if idx + 1 < len(candidates):
                next_candidate = candidates[idx + 1]
                if next_candidate["source"] == candidate["source"]:
                    next_start = next_candidate["start"] - padding
            rebuild_candidate_wav(candidate, sample_rate, padding, max_end=next_start)
        elif action in ("u", "t"):
            candidate["start"] = candidate.get("original_start", candidate["start"])
            candidate["end"] = candidate.get("original_end", candidate["end"])
            next_start = None
            if idx + 1 < len(candidates):
                next_candidate = candidates[idx + 1]
                if next_candidate["source"] == candidate["source"]:
                    next_start = next_candidate["start"] - padding
            rebuild_candidate_wav(candidate, sample_rate, padding, max_end=next_start)
        elif action == "+":
            if idx + 1 >= len(candidates):
                print("No following sample to merge with.")
            else:
                next_candidate = candidates[idx + 1]
                if next_candidate["source"] != candidate["source"]:
                    print("Cannot merge samples from different source files.")
                else:
                    history_entry = {
                        "text_before": candidate["text"],
                        "end_before": candidate["end"],
                        "original_start_before": candidate.get("original_start", candidate["start"]),
                        "original_end_before": candidate.get("original_end", candidate["end"]),
                        "next_candidate": deepcopy(next_candidate),
                    }
                    candidate["merge_history"].append(history_entry)
                    candidate["text"] = merge_forward_text(candidate["text"], next_candidate["text"])
                    candidate["end"] = next_candidate["end"]
                    candidate["original_end"] = next_candidate.get("original_end", next_candidate["end"])
                    candidate["decision"] = "pending"
                    for path_key in ("wav_path", "txt_path"):
                        try:
                            Path(next_candidate[path_key]).unlink()
                        except OSError:
                            pass
                    del candidates[idx + 1]
                    next_start = None
                    if idx + 1 < len(candidates):
                        following_candidate = candidates[idx + 1]
                        if following_candidate["source"] == candidate["source"]:
                            next_start = following_candidate["start"] - padding
                    save_candidate_text(candidate)
                    rebuild_candidate_wav(candidate, sample_rate, padding, max_end=next_start)
        elif action == "m":
            if not candidate.get("merge_history"):
                print("Nothing to unmerge.")
            else:
                history_entry = candidate["merge_history"].pop()
                next_candidate = deepcopy(history_entry["next_candidate"])
                candidate["text"] = history_entry["text_before"]
                candidate["end"] = history_entry["end_before"]
                candidate["original_start"] = history_entry["original_start_before"]
                candidate["original_end"] = history_entry["original_end_before"]
                insert_idx = idx + 1
                candidates.insert(insert_idx, next_candidate)
                ensure_candidate_metadata(next_candidate)
                save_candidate_text(candidate)
                save_candidate_text(next_candidate)
                next_start = None
                if insert_idx + 1 < len(candidates):
                    following_candidate = candidates[insert_idx + 1]
                    if following_candidate["source"] == candidate["source"]:
                        next_start = following_candidate["start"] - padding
                rebuild_candidate_wav(candidate, sample_rate, padding, max_end=next_start)
                rebuild_candidate_wav(next_candidate, sample_rate, padding)
        elif action == "\x1b":
            if confirm_yes_no("Quit and save progress?"):
                save_session(
                    session_path,
                    {
                        "language": session.get("language"),
                        "candidates": candidates,
                        "last_exported": session.get("last_exported", []),
                        "samples_dir": session.get("samples_dir"),
                        "current_index": idx,
                    },
                )
                print(f"Progress saved to {session_path}. Run with --resume to continue.")
                sys.exit(0)
            print("Continue review.")
        else:
            print("Unknown command.")
        save_session(
            session_path,
            {
                "language": session.get("language"),
                "candidates": candidates,
                "last_exported": session.get("last_exported", []),
                "samples_dir": session.get("samples_dir"),
                "current_index": idx,
            },
        )
    save_session(
        session_path,
        {
            "language": session.get("language"),
            "candidates": candidates,
            "last_exported": session.get("last_exported", []),
            "samples_dir": session.get("samples_dir"),
            "current_index": idx,
        },
    )


def append_approved_samples(
    candidates: List[Dict[str, Any]],
    samples_dir: Path,
) -> int:
    samples_dir.mkdir(parents=True, exist_ok=True)
    start_index = find_last_sample_index(samples_dir) + 1
    count = 0
    for candidate in candidates:
        if candidate["decision"] != "approved":
            continue
        out_wav = samples_dir / f"sample_{start_index}.wav"
        out_txt = samples_dir / f"sample_{start_index}.txt"
        shutil.copy2(candidate["wav_path"], out_wav)
        Path(out_txt).write_text(candidate["text"], encoding="utf-8")
        start_index += 1
        count += 1
    return count


def export_approved_samples(
    candidates: List[Dict[str, Any]],
    samples_dir: Path,
    session: Dict[str, Any],
    scratch_dir: Path,
) -> int:
    samples_dir = samples_dir.resolve()
    samples_dir.mkdir(parents=True, exist_ok=True)

    last_exported = session.get("last_exported", [])
    for file_path in last_exported:
        try:
            Path(file_path).unlink()
        except OSError:
            pass

    start_index = find_last_sample_index(samples_dir) + 1
    exported: List[str] = []
    count = 0
    for candidate in candidates:
        if candidate["decision"] != "approved":
            continue
        out_wav = samples_dir / f"sample_{start_index}.wav"
        out_txt = samples_dir / f"sample_{start_index}.txt"
        shutil.copy2(candidate["wav_path"], out_wav)
        Path(out_txt).write_text(candidate["text"], encoding="utf-8")
        exported.append(str(out_wav))
        exported.append(str(out_txt))
        start_index += 1
        count += 1

    session["last_exported"] = exported
    session["samples_dir"] = str(samples_dir)
    save_session(scratch_dir / SESSION_FILENAME, session)
    print(f"Exported {count} approved samples to {samples_dir}.")

    return count


def confirm_yes_no(prompt: str) -> bool:
    sys.stdout.write(prompt + " [y/N]: ")
    sys.stdout.flush()
    key = read_single_key()
    if key in {"\r", "\n", "\x1b", "n", "N"}:
        sys.stdout.write("\n")
        return False
    sys.stdout.write(key + "\n")
    return key.lower() == "y"


def build_session_from_audio(
    audio_paths: List[Path],
    model_name: str,
    language: Optional[str],
    sample_rate: int,
    scratch_dir: Path,
    padding: float,
    denoise: bool,
) -> Tuple[str, List[Dict[str, Any]]]:
    denoiser_obj = init_denoiser() if denoise else None
    model = load_whisper_model(model_name)
    recordings: List[Dict[str, Any]] = []
    detected_language = language
    denoised_dir = scratch_dir / "denoised"
    if denoiser_obj is not None:
        denoised_dir.mkdir(parents=True, exist_ok=True)
    for audio_path in audio_paths:
        source_path = audio_path
        if denoiser_obj is not None:
            denoised_path = denoised_dir / audio_path.name
            if not denoised_path.exists():
                with Spinner(f"Denoising: {audio_path.name}"):
                    denoise_audio(audio_path, denoised_path, denoiser_obj)
                print(f"Denoised: {audio_path.name}")
            else:
                print(f"Using cached denoised audio: {audio_path.name}")
            source_path = denoised_path
        kwargs: Dict[str, Any] = {"fp16": False}
        if detected_language:
            kwargs["language"] = detected_language
        with Spinner(f"Transcribing {source_path.name}"):
            result = model.transcribe(str(source_path), **kwargs)
        print(f"Transcribed: {source_path.name}")
        if not detected_language:
            detected_language = result.get("language", detected_language)
            if detected_language:
                print(f"Detected language: {detected_language}")
        segments = result.get("segments", [])
        sentences = build_sentence_candidates(segments)
        print(f"Detected {len(sentences)} sentence candidates in {source_path.name}")
        recordings.extend([
            {
                "audio_path": str(source_path),
                "sentence": sentence,
            }
            for sentence in sentences
        ])
    all_candidates: List[Dict[str, Any]] = []
    candidate_id = 1
    for record in recordings:
        audio_path = Path(record["audio_path"])
        sentence = record["sentence"]
        candidate = {
            "id": candidate_id,
            "text": sentence["text"],
            "wav_path": "",
            "txt_path": "",
            "decision": "pending",
            "start": sentence["start"],
            "end": sentence["end"],
            "original_start": sentence["start"],
            "original_end": sentence["end"],
            "merge_history": [],
            "source": str(audio_path),
        }
        all_candidates.append(candidate)
        candidate_id += 1
    candidate_dir = scratch_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for idx, candidate in enumerate(all_candidates):
        source = Path(candidate["source"])
        base_name = f"{CANDIDATE_PREFIX}{candidate['id']}"
        wav_path = candidate_dir / f"{base_name}.wav"
        txt_path = candidate_dir / f"{base_name}.txt"
        next_start = None
        if idx + 1 < len(all_candidates):
            next_candidate = all_candidates[idx + 1]
            if next_candidate["source"] == candidate["source"]:
                next_start = next_candidate["start"] - padding
        normalize_audio_segment(
            source,
            wav_path,
            candidate["start"],
            candidate["end"],
            sample_rate,
            padding=padding,
            max_end=next_start,
        )
        txt_path.write_text(candidate["text"], encoding="utf-8")
        candidate["wav_path"] = str(wav_path)
        candidate["txt_path"] = str(txt_path)
    return detected_language or "en", all_candidates


def human_readable_decline(candidates: List[Dict[str, Any]]) -> int:
    return sum(1 for c in candidates if c["decision"] == "approved")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sentence-level samples from audio with Whisper and append approved clips to a samples directory."
    )
    parser.add_argument("inputs", nargs="*", help="Audio files or a single directory containing audio files.")
    parser.add_argument("--samples-dir", type=Path, default=None, help="Directory to append approved sample_N.wav/.txt pairs.")
    parser.add_argument("--lang", type=str, default=None, help="Language code to force, e.g. en or pl.")
    parser.add_argument("--scratch-dir", type=Path, default=None, help="Scratch directory for temporary segments and resume state.")
    parser.add_argument("--model", type=str, default=None, help="Whisper model to use: tiny, base, small, medium, large.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Output WAV sample rate.")
    parser.add_argument("--padding", type=parse_padding, default=DEFAULT_PADDING, help="Seconds of padding to add before/after each extracted sample (minimum 0.1).")
    parser.add_argument("--denoise", action="store_true", help="Denoise extracted sentence samples using DeepFilterNet.")
    parser.add_argument("--approve-all", action="store_true", help="Auto-approve all detected samples and skip interactive review.")
    parser.add_argument("--resume", action="store_true", help="Resume a previously saved approval session in the scratch directory.")
    parser.add_argument("--recursive", action="store_true", help="Scan input directories recursively.")
    parser.add_argument("--keep-scratch", action="store_true", help="Keep scratch files after completion.")
    args = parser.parse_args()

    if args.model is None:
        args.model = detect_default_model()
    if args.scratch_dir is None:
        args.scratch_dir = Path(tempfile.mkdtemp(prefix="generate_samples_"))
    scratch_dir = args.scratch_dir.resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    session_path = scratch_dir / SESSION_FILENAME

    if args.resume:
        if not session_path.exists():
            raise SystemExit(f"No saved session found in {scratch_dir}.")
        session = load_session(session_path)
        candidates = session.get("candidates", [])
        language = session.get("language") or args.lang or "en"
        resume_idx = int(session.get("current_index", 0))
    else:
        audio_paths = find_audio_paths(args.inputs, recursive=args.recursive)
        language, candidates = build_session_from_audio(
            audio_paths,
            args.model,
            args.lang,
            args.sample_rate,
            scratch_dir,
            args.padding,
            args.denoise,
        )
        session = {
            "language": language,
            "model": args.model,
            "sample_rate": args.sample_rate,
            "candidates": candidates,
            "current_index": 0,
        }
        save_session(session_path, session)
        resume_idx = 0

    samples_dir = args.samples_dir
    if samples_dir is None:
        samples_dir = Path.cwd() / "samples" / language
    samples_dir = samples_dir.resolve()

    if args.approve_all:
        for candidate in candidates:
            candidate["decision"] = "approved"
        save_session(session_path, session)
    else:
        try:
            review_candidates(candidates, args.sample_rate, args.padding, session, scratch_dir, samples_dir, start_idx=resume_idx)
        except KeyboardInterrupt:
            save_session(session_path, {"language": session.get("language"), "candidates": candidates, "last_exported": session.get("last_exported", []), "samples_dir": session.get("samples_dir")})
            print(f"Progress saved to {session_path}. You can resume with --resume.")
            sys.exit(0)

    if args.keep_scratch:
        print(f"Scratch directory preserved at {scratch_dir}.")


if __name__ == "__main__":
    main()
