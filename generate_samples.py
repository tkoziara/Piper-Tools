#!/usr/bin/env python3
"""Generate approved sentence-level samples from audio files using Whisper."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import tty
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_SAMPLE_RATE = 22050
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3"}
SENTENCE_REGEX = re.compile(r"([^.!?]*[.!?])")
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


def normalize_audio_segment(src: Path, dst: Path, start: float, end: float, sample_rate: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ss",
        str(start),
        "-to",
        str(end),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
) -> List[Dict[str, Any]]:
    audio_base = audio_path.stem
    candidate_dir = scratch_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        base_name = f"{CANDIDATE_PREFIX}{idx}"
        wav_path = candidate_dir / f"{base_name}.wav"
        txt_path = candidate_dir / f"{base_name}.txt"
        normalize_audio_segment(audio_path, wav_path, candidate["start"], candidate["end"], sample_rate)
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


def play_audio(path: Path) -> None:
    player = None
    for prog in ("aplay", "paplay", "ffplay"):
        if shutil.which(prog):
            player = prog
            break
    if not player:
        raise SystemExit("No audio player found. Install aplay, paplay, or ffplay.")
    if player == "ffplay":
        subprocess.run([player, "-autoexit", "-nodisp", str(path)])
    else:
        subprocess.run([player, str(path)])


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


def prompt_decision() -> str:
    print("Commands: [y] approve, [n] reject, [p] play, [b] back, [f] forward, [q] quit")
    print("Choice: ", end="", flush=True)
    key = read_single_key().strip().lower()
    print(key)
    if key == "":
        return "y"
    return key[0]


def review_candidates(candidates: List[Dict[str, Any]], scratch_dir: Path) -> None:
    session_path = scratch_dir / SESSION_FILENAME
    idx = 0
    total = len(candidates)
    while idx < total:
        candidate = candidates[idx]
        print(f"\n[{idx + 1}/{total}] {candidate['text']}")
        print(f"Current decision: {candidate['decision']}")
        action = prompt_decision()
        if action == "y":
            candidate["decision"] = "approved"
            idx += 1
        elif action == "n" or action == "s":
            candidate["decision"] = "rejected"
            idx += 1
        elif action == "p":
            play_audio(Path(candidate["wav_path"]))
        elif action == "b":
            if idx > 0:
                idx -= 1
            else:
                print("Already at first sample.")
        elif action == "f":
            idx += 1
        elif action == "q":
            save_session(session_path, {"candidates": candidates})
            print(f"Progress saved to {session_path}. Run with --resume to continue.")
            sys.exit(0)
        else:
            print("Unknown command.")
        save_session(session_path, {"candidates": candidates})
    save_session(session_path, {"candidates": candidates})


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


def confirm_yes_no(prompt: str) -> bool:
    answer = input(prompt + " [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def build_session_from_audio(
    audio_paths: List[Path],
    model_name: str,
    language: Optional[str],
    sample_rate: int,
    scratch_dir: Path,
) -> Tuple[str, List[Dict[str, Any]]]:
    model = load_whisper_model(model_name)
    recordings: List[Dict[str, Any]] = []
    detected_language = language
    for audio_path in audio_paths:
        print(f"Transcribing: {audio_path.name}")
        kwargs: Dict[str, Any] = {"fp16": False}
        if detected_language:
            kwargs["language"] = detected_language
        result = model.transcribe(str(audio_path), **kwargs)
        if not detected_language:
            detected_language = result.get("language", detected_language)
            if detected_language:
                print(f"Detected language: {detected_language}")
        segments = result.get("segments", [])
        sentences = build_sentence_candidates(segments)
        print(f"Detected {len(sentences)} sentence candidates in {audio_path.name}")
        recordings.extend([
            {
                "audio_path": str(audio_path),
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
            "source": str(audio_path),
        }
        all_candidates.append(candidate)
        candidate_id += 1
    candidate_dir = scratch_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for candidate in all_candidates:
        source = Path(candidate["source"])
        base_name = f"{CANDIDATE_PREFIX}{candidate['id']}"
        wav_path = candidate_dir / f"{base_name}.wav"
        txt_path = candidate_dir / f"{base_name}.txt"
        normalize_audio_segment(source, wav_path, candidate["start"], candidate["end"], sample_rate)
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
        language = session.get("language", args.lang)
    else:
        audio_paths = find_audio_paths(args.inputs, recursive=args.recursive)
        language, candidates = build_session_from_audio(
            audio_paths,
            args.model,
            args.lang,
            args.sample_rate,
            scratch_dir,
        )
        session = {
            "language": language,
            "model": args.model,
            "sample_rate": args.sample_rate,
            "candidates": candidates,
        }
        save_session(session_path, session)

    if args.approve_all:
        for candidate in candidates:
            candidate["decision"] = "approved"
        save_session(session_path, session)
    else:
        try:
            review_candidates(candidates, scratch_dir)
        except KeyboardInterrupt:
            save_session(session_path, {"language": session.get("language"), "candidates": candidates})
            print(f"Progress saved to {session_path}. You can resume with --resume.")
            sys.exit(0)

    samples_dir = args.samples_dir
    if samples_dir is None:
        samples_dir = Path.cwd() / "samples" / language
    samples_dir = samples_dir.resolve()
    appended = append_approved_samples(candidates, samples_dir)
    print(f"Appended {appended} approved samples to {samples_dir}.")

    if appended and not args.keep_scratch:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        print(f"Removed scratch directory {scratch_dir}.")
    elif args.keep_scratch:
        print(f"Scratch directory preserved at {scratch_dir}.")


if __name__ == "__main__":
    main()
