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


def ensure_candidate_metadata(candidate: Dict[str, Any]) -> None:
    if "original_start" not in candidate:
        candidate["original_start"] = candidate["start"]
    if "original_end" not in candidate:
        candidate["original_end"] = candidate["end"]
    if "merge_history" not in candidate:
        candidate["merge_history"] = []


def init_denoiser() -> tuple:
    try:
        from df.enhance import init_df
    except ImportError as exc:
        raise SystemExit(
            "Denoising requested, but DeepFilterNet is not installed. "
            "Install it with `pip install deepfilternet`."
        ) from exc
    model, df_state, _ = init_df()
    return model, df_state


def denoise_audio(src: Path, dst: Path, denoiser: Optional[tuple]) -> None:
    if denoiser is None:
        shutil.copy2(src, dst)
        return
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


def prompt_decision(candidate: Optional[Dict[str, Any]] = None) -> str:
    commands = [
        "[y] approve",
        "[n] reject",
        "[p] play",
        "[b] back",
        "[f] forward",
        "[q] quit",
        "[e] edit text",
        "[-] step back",
        "[=] trim end",
        "[r] restore trims",
        "[+] merge forward",
    ]
    if candidate and candidate.get("merge_history"):
        commands.append("[u] unmerge")
    print("Commands: " + ", ".join(commands))
    print("Choice: ", end="", flush=True)
    key = read_single_key().strip().lower()
    print(key)
    if key == "":
        return "y"
    return key[0]


def review_candidates(
    candidates: List[Dict[str, Any]],
    sample_rate: int,
    padding: float,
) -> None:
    for candidate in candidates:
        ensure_candidate_metadata(candidate)
    session_path = Path(candidates[0]["wav_path"]).parent.parent / SESSION_FILENAME if candidates else Path(SESSION_FILENAME)
    idx = 0
    while idx < len(candidates):
        candidate = candidates[idx]
        total = len(candidates)
        print(f"\n[{idx + 1}/{total}] {candidate['text']}")
        print(f"Current decision: {candidate['decision']}")
        action = prompt_decision(candidate)
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
        elif action == "e":
            print(f"Current text: {candidate['text']}")
            edited = input_with_prefill("New text (leave blank to keep current): ", candidate["text"]).strip()
            if edited and edited != candidate["text"]:
                candidate["text"] = edited
                save_candidate_text(candidate)
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
        elif action == "r":
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
        elif action == "u":
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
                denoise_audio(audio_path, denoised_path, denoiser_obj)
            source_path = denoised_path
        print(f"Transcribing: {source_path.name}")
        kwargs: Dict[str, Any] = {"fp16": False}
        if detected_language:
            kwargs["language"] = detected_language
        result = model.transcribe(str(source_path), **kwargs)
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
        language = session.get("language", args.lang)
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
        }
        save_session(session_path, session)

    if args.approve_all:
        for candidate in candidates:
            candidate["decision"] = "approved"
        save_session(session_path, session)
    else:
        try:
            review_candidates(candidates, args.sample_rate, args.padding)
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
