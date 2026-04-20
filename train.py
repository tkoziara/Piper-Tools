#!/usr/bin/env python3
"""Simplified Piper training runner.

Commands implemented:
- `init` / `prepare`: normalize audio, build `wavs/` and `metadata.csv` ready for Piper training.
- `fetch-base`: download a Piper checkpoint (from Hugging Face `rhasspy/piper-checkpoints`) if needed.
- `train`: print or run `python3 -m piper.train fit` using sensible quality presets.
- `export`: export a trained checkpoint to ONNX using Piper's export helper.
- `synth-test`: synthesize a test WAV from an exported ONNX model using `python3 -m piper`.

This script will attempt to `pip install huggingface_hub` in the active venv
when `fetch-base` is used and the package is not available.

Defaults and safety:
- Sample rate: 22050 Hz.
- `train` prints the command by default; use `--run` to execute.
- Quote normalization: various quotation styles („...", "...", «...») are normalized to ASCII "..." by default.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import tempfile
import json
import warnings

# Suppress a noisy FutureWarning emitted by some versions of torch when
# libraries call `torch.load`.  Keep the filter targeted to FutureWarning
# messages referencing `torch.load` so other useful warnings still appear.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

SAMPLE_RATE = 22050
SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def find_audio_files(dirpath: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(dirpath.rglob("*")):
        if p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            files.append(p)
    return files


def normalize_audio(src: Path, dst: Path, sample_rate: int = SAMPLE_RATE) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def normalize_quotes(text: str) -> str:
    """Normalize various quotation mark styles to ASCII double quotes.

    Handles:
    - Polish/curly quotes: „ ... "
    - English curly quotes: " ... "
    - French quotes: « ... »
    - Single curly quotes: ' ... '
    - Reversed Polish quotes: « ... " (treated as opening)
    """
    # Map of opening/closing quote pairs to ASCII double quotes
    # Order matters: process paired quotes first, then stray singles
    replacements = [
        ("„", '"'),  # Polish low-high
        ("'", '"'),  # Polish/curly high
        ("'", '"'),  # Curly single close
        ("'", '"'),  # Curly single open
        ("«", '"'),  # French/Guillemet open
        ("»", '"'),  # French/Guillemet close
        ("‹", '"'),  # Single guillemet open
        ("›", '"'),  # Single guillemet close
        ("‟", '"'),  # High-6 quote
        ("‛", '"'),  # Single high-reversed
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def collect_transcript_for_audio(audio_path: Path) -> str:
    """Look for transcript files next to the audio or a global transcripts.txt.

    If none are found, fall back to the filename (underscores -> spaces) as a placeholder.

    Quote normalization is always applied: various quotation styles („...", "...", «...»)
    are converted to ASCII double quotes ("...").
    """
    # 1) same-name .txt
    txt_same = audio_path.with_suffix(".txt")
    if txt_same.exists():
        text = txt_same.read_text(encoding="utf-8").strip()
        return normalize_quotes(text)

    # 2) transcripts.txt in same directory
    transcripts = audio_path.parent / "transcripts.txt"
    if transcripts.exists():
        for line in transcripts.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # allow either: filename|transcript or filename transcript
            if "|" in line:
                name, text = line.split("|", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    name, text = parts
                else:
                    continue
            if Path(name).stem == audio_path.stem:
                return normalize_quotes(text.strip())

    # fallback: create readable text from filename
    return audio_path.stem.replace("_", " ").replace("-", " ")


def prepare_dataset(samples_dir: Path, out_dir: Path, lang: str, quality: str, no_convert: bool = False, config_name: Optional[str] = None) -> None:
    samples_dir = samples_dir.resolve()
    out_dir = out_dir.resolve()
    if not samples_dir.exists():
        raise SystemExit(f"Samples directory not found: {samples_dir}")

    # Allow `--samples-dir` to simply point to the directory with samples in given language
    if samples_dir.exists():
        lang_dir = samples_dir
    else:
        raise SystemExit(f"Language samples directory not found: {lang_dir}")

    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(lang_dir)
    if not audio_files:
        raise SystemExit(f"No audio files found under {lang_dir}")

    metadata_path = out_dir / "metadata.csv"
    rows: List[Tuple[str, str]] = []

    print(f"Preparing {len(audio_files)} files from {lang_dir} -> {wavs_dir}")

    for i, src in enumerate(audio_files, start=1):
        out_fname = f"{lang}_{i:04d}.wav"
        dst = wavs_dir / out_fname
        if not no_convert:
            normalize_audio(src, dst)
        else:
            # copy without conversion
            shutil.copy2(src, dst)

        transcript = collect_transcript_for_audio(src)
        rows.append((out_fname, transcript))

    # write metadata.csv as: filename|transcript (paths are relative to the audio_dir)
    # Use CSV writer with proper escaping for quotes (required by Piper's csv.reader)
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for fname, text in rows:
            writer.writerow([fname, text])

    # save a small README with training suggestions
    readme = out_dir / "README_TRAINING.md"
    with readme.open("w", encoding="utf-8") as f:
        f.write(r"""
This dataset has been prepared for Piper training.

Contents:
- wavs/          : normalized WAV files (22050 Hz, mono, 16‑bit)
- metadata.csv   : lines `xxx.wav|transcript` (paths are relative to the
  `wavs/` directory; **do not** include the `wavs/` prefix).

You can now fine‑tune a Piper model using this folder.  Two convenient ways are
provided in this repository:

* run the helper script (preferred):

    ./train_and_export.sh --out-dir <path> --voice-name <name> \
        [--quality medium] [--epochs N] [--rounds R] [--yes]

  this will fetch a base checkpoint if necessary, execute training (optionally),
  export ONNX models after each round and synth‑test a short sample.

* call the `train.py` commands directly, e.g.:

    python3 train.py train --out-dir <path> --ckpt ~/.piper/checkpoints/medium/base_checkpoint.ckpt --voice-name <name> [--run]
    python3 train.py export --checkpoint <ckpt> --output <file>.onnx
    python3 train.py synth-test --model <file>.onnx --text "..."

See the repository's `TRAIN.md` document for full details and troubleshooting.
""")

    print("Dataset prepared:", out_dir)
    print("Metadata saved to:", metadata_path)
    print("See README_TRAINING.md inside the dataset folder for suggested next steps.")

    # Write a minimal voice_config.json to make training easier to run
    voice_config = {
        "data": {
            "voice_name": f"{lang}_custom",
            "sample_rate": SAMPLE_RATE,
            "espeak_voice": "en-us" if lang == "en" else "pl",
            "csv_path": str(metadata_path.name),
            "audio_dir": "wavs"
        },
        "model": {
            "sample_rate": SAMPLE_RATE
        },
        "notes": "Minimal config generated by train.py. Edit for advanced options."
    }
    config_basename = config_name or "voice_config"
    config_path = out_dir / f"{config_basename}.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(voice_config, f, indent=2)
    print("Wrote minimal voice config to:", config_path)


def quality_presets(quality: str) -> Dict[str, str]:
    q = quality.lower()
    if q == "low":
        return {"epochs": "50", "batch_size": "32", "notes": "fast, low quality"}
    if q == "high":
        return {"epochs": "400", "batch_size": "8", "notes": "high quality, slow"}
    # default medium
    return {"epochs": "200", "batch_size": "16", "notes": "balanced"}


def ensure_hf_hub_installed() -> None:
    try:
        import huggingface_hub  # type: ignore
        return
    except Exception:
        print("`huggingface_hub` not found — installing into current venv...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)


def fetch_base_checkpoint(dest_dir: Path, quality: str = "medium", model_id: Optional[str] = None, model_pattern: Optional[str] = None, yes: bool = False) -> Path:
    """Download a base checkpoint from Hugging Face `rhasspy/piper-checkpoints`.

    This function first attempts a targeted single-file download using the
    `huggingface_hub` API (fast, small). It lists files in the `rhasspy/piper-checkpoints`
    dataset, filters `.ckpt` files by `model_id` or `model_pattern` and downloads
    the selected file with `hf_hub_download`. If no candidate is found, it falls
    back to `snapshot_download` (full dataset snapshot).

    Parameters:
    - dest_dir: where the checkpoint will be copied/placed
    - quality: hint used to prioritize filenames containing the quality string
    - model_id: optional filename or model id to fetch directly
    - model_pattern: substring or regex to match filenames (e.g. 'en', 'pl', 'male')
    - yes: non-interactive; accept first match

    Returns the path to the downloaded checkpoint file.
    """
    dest_dir = dest_dir.expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    ensure_hf_hub_installed()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # type: ignore

    repo = "rhasspy/piper-checkpoints"

    api = HfApi()
    print(f"Querying {repo} for available checkpoint files...")
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    except Exception as e:
        print("Could not list repo files (network or rate limit?). Falling back to full snapshot.")
        files = []

    # filter ckpt files
    ckpt_files = [f for f in files if f.endswith(".ckpt")]

    candidate: Optional[str] = None
    candidates: List[str] = []
    if model_id:
        # if model_id is a filename present in the repo, use it directly
        if model_id in ckpt_files:
            candidates = [model_id]
        else:
            # allow model_id to be a path-like substring
            candidates = [f for f in ckpt_files if model_id in f]
    elif model_pattern:
        candidates = [f for f in ckpt_files if model_pattern in f]
    else:
        # prefer files that have the quality in their name
        for f in ckpt_files:
            if quality in f:
                candidates.append(f)
        # fallback to all ckpt files
        if not candidates:
            candidates = ckpt_files

    if candidates:
        # de-duplicate and sort
        seen = set()
        uniq = []
        for c in candidates:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        candidates = uniq

        print(f"Found {len(candidates)} candidate checkpoint(s).")
        # if multiple candidates and not yes, interactively ask the user
        # If a model_pattern was requested, prioritize matches containing it (case-insensitive)
        if model_pattern:
            pat = model_pattern.lower()
            matched = [c for c in candidates if pat in c.lower()]
            other = [c for c in candidates if pat not in c.lower()]
            if matched:
                candidates = matched + other

        if len(candidates) == 1 or yes:
            candidate = candidates[0]
            print(f"Selected candidate: {candidate}")
        else:
            print("Candidates (showing all matches; choose a number):")
            # show up to 500 candidates to ensure pl entries are visible
            for i, c in enumerate(candidates[:500], start=1):
                print(f"  {i}. {c}")
            try:
                choice = input("Select file number to download (or blank to cancel): ")
            except Exception:
                choice = ""
            if not choice.strip():
                print("No selection made — falling back to full snapshot download.")
                candidate = None
            else:
                try:
                    idx = int(choice.strip()) - 1
                    if 0 <= idx < len(candidates):
                        candidate = candidates[idx]
                    else:
                        print("Invalid choice — falling back to full snapshot")
                        candidate = None
                except Exception:
                    print("Invalid input — falling back to full snapshot")
                    candidate = None

    if candidate:
        print(f"Downloading single checkpoint file: {candidate}")
        try:
            downloaded = hf_hub_download(repo_id=repo, filename=candidate, repo_type="dataset", cache_dir=str(dest_dir))
            target = dest_dir / "base_checkpoint.ckpt"
            shutil.copy2(downloaded, target)
            print(f"Downloaded checkpoint to: {target}")
            return target
        except Exception as e:
            print(f"Failed to download single file {candidate}: {e}")
            print("Falling back to full snapshot download...")

    # Fallback: snapshot the full dataset (may be multi-GB)
    print(f"Snapshotting the {repo} dataset (may be large)...")
    snapshot_path = snapshot_download(repo_id=repo, cache_dir=str(dest_dir), repo_type="dataset")
    ckpt_files2 = list(Path(snapshot_path).rglob("*.ckpt"))
    if not ckpt_files2:
        raise SystemExit("No .ckpt files found in the checkpoint snapshot")
    preferred = None
    for p in ckpt_files2:
        if quality in p.name:
            preferred = p
            break
    if preferred is None:
        preferred = ckpt_files2[0]

    target = dest_dir / "base_checkpoint.ckpt"
    shutil.copy2(preferred, target)
    print(f"Downloaded snapshot checkpoint to: {target}")
    return target


def build_train_command(out_dir: Path, ckpt_path: Path, voice_name: str, quality: str, gpu: bool, epochs: int | None = None, batch_size: int | None = None, num_workers: int | None = None, espeak_voice: Optional[str] = None, config_path: Optional[Path] = None) -> List[str]:
    preset = quality_presets(quality)
    if epochs is not None:
        # user override; adjust preset value so that later code uses it
        preset = dict(preset)
        preset["epochs"] = epochs
    if batch_size is not None:
        # override the preset batch size
        preset = dict(preset)
        preset["batch_size"] = str(batch_size)
    out_dir = out_dir.resolve()
    # ensure training outputs (logs & checkpoints) land inside the dataset
    # folder so our wrapper can find them.  Lightning respects the
    # ``--trainer.default_root_dir`` option which affects both logger and
    # checkpoint callback locations.
    tts_dir = out_dir / "tts_output"

    # determine espeak voice to pass through
    if espeak_voice is None:
        # if caller didn't override, try to read from a config file
        cfg_to_read = None
        if config_path is not None:
            cfg_to_read = config_path
        else:
            cfg_to_read = out_dir / "voice_config.json"
        try:
            if cfg_to_read and cfg_to_read.exists():
                with cfg_to_read.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    es = cfg.get("data", {}).get("espeak_voice")
                    if es:
                        espeak_voice = es
        except Exception:
            espeak_voice = None
        if not espeak_voice:
            espeak_voice = "en-us"
    # build argument list for `piper.train fit`
    args: List[str] = [
        "fit",
        "--trainer.default_root_dir",
        str(tts_dir),
        "--data.voice_name",
        voice_name,
        "--data.csv_path",
        str(out_dir / "metadata.csv"),
        "--data.audio_dir",
        str(out_dir / "wavs"),
        "--model.sample_rate",
        str(SAMPLE_RATE),
        "--data.cache_dir",
        str(out_dir / "cache"),
        "--data.config_path",
        str(config_path if config_path is not None else (out_dir / "voice_config.json")),
        "--data.batch_size",
        str(preset["batch_size"]),
        "--data.num_workers",
        str(num_workers) if num_workers is not None else "0",
        "--data.espeak_voice",
        str(espeak_voice),
        "--ckpt_path",
        str(ckpt_path),
        "--trainer.max_epochs",
        str(preset["epochs"]),
    ]
    if gpu:
        args += ["--device", "cuda"]
    return args

    if gpu:
        cmd.append("--device")
        cmd.append("cuda")
    return cmd


def sanitize_checkpoint(ckpt_path: Path) -> Path:
    """Return a path to a checkpoint file suitable for CLI loading.

    Some checkpoints (base checkpoints from HF) include hyperparameters that
    Lightning's CLI will try to parse as command-line options.  In particular,
    ``model.sample_bytes`` is not recognized and causes a parse failure.  We
    load the checkpoint with ``weights_only=False`` (to read the hyperparameters),
    strip any offending entries, and save a cleaned copy alongside the original
    (``*.clean.ckpt``).  Subsequent calls reuse the cleaned file.
    """
    import torch, pathlib

    # ensure safe globals for pathlib if supported (PyTorch version compatibility)
    try:
        torch.serialization.add_safe_globals([pathlib.PosixPath])
    except AttributeError:
        pass

    orig = ckpt_path.resolve()
    # If the checkpoint already appears to be a cleaned file, return it
    # unchanged to avoid repeatedly appending `.clean.ckpt`.
    if orig.name.endswith('.clean.ckpt'):
        return orig
    clean = orig.with_suffix(orig.suffix + ".clean.ckpt")
    # attempt to load original checkpoint regardless of whether a cleaned file
    # already exists; we want to overwrite the cleaned file with the latest
    # stripped version.
    try:
        data = torch.load(str(orig), map_location="cpu", weights_only=False)
    except Exception:
        return orig
    if isinstance(data, dict):
        changed = False
        if "hyper_parameters" in data or "hparams" in data:
            print("Sanitizing checkpoint: stripping hyperparameters")
            data.pop("hyper_parameters", None)
            data.pop("hparams", None)
            changed = True
        # reset epoch/global_step so Trainer doesn't think we're already done
        if "epoch" in data and data.get("epoch", 0) > 0:
            print("Resetting checkpoint epoch to 0")
            data["epoch"] = 0
            changed = True
        if "global_step" in data and data.get("global_step", 0) > 0:
            data["global_step"] = 0
            changed = True
        # drop training loops but keep optimizer/scheduler keys empty so
        # Lightning CLI doesn't raise when attempting to restore them.
        if "loops" in data:
            print("Removing training state 'loops' from checkpoint")
            data.pop("loops", None)
            changed = True
        # include empty optimizer_states/lr_schedulers to satisfy CLI restore logic
        if "optimizer_states" in data:
            print("Clearing optimizer_states in checkpoint")
            data["optimizer_states"] = []
            changed = True
        else:
            # add empty anyway for safety
            data["optimizer_states"] = []
            changed = True
        if "lr_schedulers" in data:
            print("Clearing lr_schedulers in checkpoint")
            data["lr_schedulers"] = []
            changed = True
        else:
            data["lr_schedulers"] = []
            changed = True
        if changed:
            torch.save(data, str(clean))
            return clean
    # nothing to remove; just return original path
    return orig


def run_train(out_dir: Path, ckpt_path: Path, voice_name: str, quality: str, gpu: bool, run: bool, epochs: int | None = None, batch_size: int | None = None, num_workers: int | None = None, espeak_voice: Optional[str] = None) -> None:
    # sanitize checkpoint to avoid CLI hyperparam parsing issues
    ckpt_path = sanitize_checkpoint(ckpt_path)

    # figure out a safe dataset config file to supply to Piper.  Training
    # may overwrite whichever config path is given, so we copy the original
    # (or a backup) to a temporary file when actually running.
    cfg_orig = out_dir / "voice_config.dataset.json"
    if not cfg_orig.exists():
        cfg_orig = out_dir / "voice_config.json"
    cfg_to_use: Optional[Path]
    if cfg_orig.exists():
        if run:
            tmp = Path(tempfile.mktemp(suffix=".json"))
            # copy original dataset config so modifications don't clobber it
            shutil.copy2(cfg_orig, tmp)
            cfg_to_use = tmp
            # if an explicit espeak override was requested, apply it to temp
            if espeak_voice:
                try:
                    j = json.load(tmp.open("r", encoding="utf-8"))
                    j.setdefault("data", {})["espeak_voice"] = espeak_voice
                    json.dump(j, tmp.open("w", encoding="utf-8"), indent=2)
                except Exception:
                    pass
        else:
            cfg_to_use = cfg_orig
    else:
        cfg_to_use = None

    args = build_train_command(out_dir, ckpt_path, voice_name, quality, gpu, epochs, batch_size, num_workers, espeak_voice=espeak_voice, config_path=cfg_to_use)
    full_cmd = [sys.executable, "-m", "piper.train"] + args
    print("Training command:")
    print(" ".join(full_cmd))
    if run:
        print("Executing training command (this may take a long time)...")
        # run with a small Python snippet that adds torch safe globals before
        # the module is imported (needed for checkpoints containing Pathlib)
        snippet = f"""import torch, pathlib, runpy, sys
try:
    torch.serialization.add_safe_globals([pathlib.PosixPath])
except AttributeError:
    pass
sys.argv = ['piper.train'] + {args!r}
runpy.run_module('piper.train', run_name='__main__')
"""
        subprocess.run([sys.executable, "-c", snippet], check=True)


def export_onnx(checkpoint: Path, output_file: Path, espeak_voice: Optional[str] = None) -> None:
    output_file = output_file.resolve()
    if output_file.suffix.lower() != ".onnx":
        output_file = output_file.with_suffix(".onnx")
    cmd = [sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(checkpoint), "--output-file", str(output_file)]
    print("Export command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # ensure a corresponding model config JSON exists; if the exporter did
    # not produce one, synth_test and piper will fail, so generate a minimal
    # fallback using hyperparameters from the checkpoint.
    json_path = output_file.with_suffix(output_file.suffix + ".json")

    def _json_invalid(path: Path) -> bool:
        if not path.exists():
            return True
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return True
        if not isinstance(data, dict):
            return True
        phoneme_id_map = data.get("phoneme_id_map")
        return not isinstance(phoneme_id_map, dict) or len(phoneme_id_map) == 0

    if _json_invalid(json_path):
        print(f"Exporter did not create a valid config at {json_path}; generating fallback model config.")
        try:
            import torch
            from piper.config import PiperConfig, PhonemeType
            from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP

            def _find_dataset_config(cp: Path) -> Path | None:
                current = cp.parent
                for _ in range(5):
                    for candidate in ("voice_config.json", "voice_config.dataset.json"):
                        candidate_path = current / candidate
                        if candidate_path.exists():
                            return candidate_path
                    if current.parent == current:
                        break
                    current = current.parent

                stem = cp.stem
                if stem.endswith("-latest"):
                    candidate_path = cp.parent / stem[:-7] / "voice_config.json"
                    if candidate_path.exists():
                        return candidate_path
                return None

            data = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            hp: dict = {}
            if isinstance(data, dict):
                hp = data.get("hyper_parameters", {}) or data.get("hparams", {}) or {}

            config_file = _find_dataset_config(checkpoint)
            config_map = None
            if config_file is not None:
                try:
                    config_map = json.load(open(config_file, "r", encoding="utf-8"))
                except Exception:
                    config_map = None

            espeak = espeak_voice
            phoneme_type = PhonemeType.ESPEAK
            phoneme_id_map = None
            if config_map is not None and isinstance(config_map, dict):
                espeak = (
                    config_map.get("data", {}).get("espeak_voice")
                    or config_map.get("espeak", {}).get("voice")
                    or espeak
                )
                phoneme_type_value = config_map.get("phoneme_type")
                if phoneme_type_value:
                    try:
                        phoneme_type = PhonemeType(phoneme_type_value)
                    except ValueError:
                        phoneme_type = PhonemeType.ESPEAK
                map_from_cfg = config_map.get("phoneme_id_map")
                if isinstance(map_from_cfg, dict) and map_from_cfg:
                    phoneme_id_map = map_from_cfg

            if not espeak:
                espeak = "pl" if "pl" in str(checkpoint) else "en-us"

            if phoneme_id_map is None:
                if phoneme_type == PhonemeType.PINYIN:
                    from piper.phonemize_chinese import PHONEME_TO_ID

                    phoneme_id_map = PHONEME_TO_ID
                else:
                    phoneme_id_map = DEFAULT_PHONEME_ID_MAP

            cfg_obj = PiperConfig(
                num_symbols=hp.get("num_symbols", 256),
                num_speakers=hp.get("num_speakers", 1),
                sample_rate=hp.get("sample_rate", SAMPLE_RATE),
                espeak_voice=espeak,
                phoneme_id_map=phoneme_id_map,
                phoneme_type=phoneme_type,
            )
            cfg = cfg_obj.to_dict()
            cfg.setdefault("phoneme_map", {})
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print("Wrote fallback model config to", json_path)
        except Exception as e:
            print("Warning: failed to create fallback model config:", e)


def synth_test(model: Path, text: str, out_file: Path) -> None:
    model = model.resolve()
    out_file = out_file.resolve()
    if model.is_dir():
        model_dir = model
        # find .onnx inside
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise SystemExit(f"No .onnx file found in {model_dir}")
        onnx_file = onnx_files[0]
    else:
        if model.suffix == ".onnx":
            onnx_file = model
            # prepare temporary directory with model + config
            tmp = Path(tempfile.mkdtemp(prefix="piper_synth_"))
            shutil.copy2(str(model), str(tmp / model.name))

            # locate config: prefer real model JSON; if missing, try dataset
            json_conf = model.with_suffix(model.suffix + ".json")
            if not json_conf.exists():
                # look for dataset config backup in same folder
                dsbackup = model.parent / "voice_config.dataset.json"
                if dsbackup.exists():
                    print("Model JSON missing; copying dataset backup for synth.")
                    json_conf = dsbackup
            if json_conf.exists():
                shutil.copy2(str(json_conf), str(tmp / json_conf.name))
                # validate config
                try:
                    cfg = json.load(open(tmp / json_conf.name))
                except Exception:
                    raise SystemExit(f"Invalid JSON model config: {json_conf}")
                if 'num_symbols' not in cfg:
                    raise SystemExit(
                        f"Model JSON {json_conf} does not appear to be a model config (missing 'num_symbols').\n"
                        "Please re-export the ONNX with a proper model config."
                    )
            else:
                # try to auto-create minimal config from checkpoint if available
                ckpt_guess = model.parent / "tts_output" / "lightning_logs" / "version_0" / "checkpoints" / "epoch=0-step=0.ckpt"
                if ckpt_guess.exists():
                    print("No JSON config found; attempting to generate one from checkpoint")
                    try:
                        import torch
                        d = torch.load(str(ckpt_guess), map_location='cpu', weights_only=False)
                        hp = d.get('hyper_parameters', {}) or d.get('hparams', {}) or {}
                        cfg = {'num_symbols': hp.get('num_symbols', 256)}
                        cfg.setdefault('model', {})['sample_rate'] = hp.get('sample_rate', SAMPLE_RATE)
                        with open(tmp / (model.stem + '.onnx.json'), 'w') as f:
                            json.dump(cfg, f, indent=2)
                        print("Generated fallback JSON at", tmp / (model.stem + '.onnx.json'))
                    except Exception:
                        pass
            model_dir = tmp
            onnx_file = tmp / model.name
        else:
            raise SystemExit("Model path must be a directory or an .onnx file")

    # derive model name for piper -m argument (basename without .onnx)
    model_name = onnx_file.stem
    cmd = [sys.executable, "-m", "piper", "-m", model_name, "--data-dir", str(model_dir), "-f", str(out_file), "--", text]
    print("Synth command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def synth_test_checkpoint(checkpoint: Path, text: str, out_file: Path, espeak_voice: str = "en-us") -> None:
    checkpoint = checkpoint.resolve()
    out_file = out_file.resolve()

    import torch
    from piper.phoneme_ids import phonemes_to_ids
    from piper.phonemize_espeak import EspeakPhonemizer
    from piper.train.vits.lightning import VitsModel

    print(f"Synthesizing from checkpoint: {checkpoint}")
    model = VitsModel.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()

    model_g = model.model_g
    with torch.no_grad():
        try:
            model_g.dec.remove_weight_norm()
        except Exception:
            pass

        phonemizer = EspeakPhonemizer()
        phoneme_lists = phonemizer.phonemize(espeak_voice, text)
        phonemes = [p for sentence in phoneme_lists for p in sentence]
        ids = phonemes_to_ids(phonemes)

        if len(ids) == 0:
            raise ValueError("No phonemes generated for text")

        text_tensor = torch.LongTensor(ids).unsqueeze(0)
        text_lengths = torch.LongTensor([text_tensor.size(1)])

        scales = torch.FloatTensor([0.667, 1.0, 0.8])

        sid = None
        try:
            n_speakers = model.model_g.n_speakers
        except Exception:
            n_speakers = 1

        if n_speakers > 1:
            sid = torch.LongTensor([0])

        out_audio, _, _, _ = model_g.infer(
            text_tensor,
            text_lengths,
            sid=sid,
            noise_scale=scales[0],
            length_scale=scales[1],
            noise_scale_w=scales[2],
        )

    out_audio = out_audio.squeeze().cpu().numpy()
    out_audio = (out_audio * 32767.0).clip(-32768, 32767).astype('int16')

    import wave

    sample_rate = 22050
    # model_g may expose sample_rate directly or via a .hparams object/dict
    if hasattr(model.model_g, 'sample_rate'):
        sample_rate = model.model_g.sample_rate
    else:
        hparams_obj = getattr(model.model_g, 'hparams', None) or getattr(model, 'hparams', None)
        if hparams_obj is not None:
            try:
                sample_rate = int(getattr(hparams_obj, 'sample_rate', hparams_obj.get('sample_rate', SAMPLE_RATE)))
            except Exception:
                sample_rate = SAMPLE_RATE

    with wave.open(str(out_file), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(out_audio.tobytes())

    print(f"Wrote checkpoint synth output to {out_file}")


def print_training_command(out_dir: Path, backend: str, quality: str, gpu: bool, run: bool) -> None:
    out_dir = out_dir.resolve()
    preset = quality_presets(quality)
    print("\nTraining backend:", backend)
    print("Quality preset:", quality, preset)

    if backend.lower() == "coqui":
        cmd = (
            "# Example Coqui TTS training command (edit paths and config as needed)\n"
            f"python3 TTS/bin/train.py --continue_path {out_dir / 'tts_output'} \\\n+--config_path {out_dir / 'coqui_config.json'} --datasets "
            f"{out_dir}"
        )
        print(cmd)
        if run:
            print("\nRunning the example training command now (this will shell out).")
            subprocess.run(cmd, shell=True, check=True)
    else:
        print("No automated training implemented for backend:", backend)
        print("Please follow your chosen toolkit's docs and point it at:", out_dir)


def main():
    parser = argparse.ArgumentParser(prog="train.py", description="Prepare and help run TTS training for samples.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # prepare / init (alias)
    p_prepare = sub.add_parser("init", help="Prepare samples into a training dataset (alias: prepare)")
    p_prepare.add_argument("--samples-dir", type=Path, default=Path("samples"), help="Root samples directory")
    p_prepare.add_argument("--out-dir", type=Path, default=Path("data"), help="Output dataset folder")
    p_prepare.add_argument("--lang", choices=["en", "pl"], required=True, help="Language subfolder to use")
    p_prepare.add_argument("--quality", choices=["low", "medium", "high"], default="medium", help="Quality preset (affects suggested training params)")
    p_prepare.add_argument("--voice-name", type=str, default=None, help="Optional output config basename; writes <name>.json instead of voice_config.json")
    p_prepare.add_argument("--no-convert", action="store_true", help="Do not run ffmpeg conversion; copy files instead")

    p_prepare2 = sub.add_parser("prepare", help=argparse.SUPPRESS)
    p_prepare2._action_groups = p_prepare._action_groups

    # fetch-base
    p_fetch = sub.add_parser("fetch-base", help="Download a base Piper checkpoint from Hugging Face")
    p_fetch.add_argument("--dest-dir", type=Path, default=Path.home() / ".piper" / "checkpoints", help="Destination directory for checkpoint")
    p_fetch.add_argument("--quality", choices=["low", "medium", "high"], default="medium")
    p_fetch.add_argument("--model-id", type=str, default="", help="Optional model id or filename to fetch")
    p_fetch.add_argument("--model-pattern", type=str, default="", help="Regex or substring to match checkpoint filenames (e.g. 'en' or 'pl' or 'male')")
    p_fetch.add_argument("--yes", action="store_true", help="Non-interactive: accept first match")

    # train
    p_train = sub.add_parser("train", help="Print (or run) Piper training command (fine-tune)")
    p_train.add_argument("--out-dir", type=Path, default=Path("data"), help="Prepared dataset folder")
    p_train.add_argument("--ckpt", type=Path, required=False, help="Base checkpoint path (if omitted, fetch-base will be attempted)")
    p_train.add_argument("--voice-name", type=str, default="voice", help="Name for the voice to write into config")
    p_train.add_argument("--quality", choices=["low", "medium", "high"], default="medium")
    p_train.add_argument("--epochs", type=int, default=None,
                         help="Override number of training epochs (useful for quick tests")
    p_train.add_argument("--gpu", action="store_true", help="Indicate GPU available")
    p_train.add_argument("--batch-size", type=int, default=None,
                         help="Override batch size (otherwise derived from quality preset)")
    p_train.add_argument("--num-workers", type=int, default=None,
                         help="Override DataLoader num_workers (default: auto / set by wrapper)")
    p_train.add_argument("--espeak-voice", type=str, default=None,
                         help="Override espeak voice (e.g. 'pl' or 'en-us').  The wrapper will set this automatically based on language.")
    p_train.add_argument("--run", action="store_true", help="Execute the training command")

    # export
    p_export = sub.add_parser("export", help="Export a trained checkpoint to ONNX")
    p_export.add_argument("--checkpoint", type=Path, required=True, help="Path to a .ckpt checkpoint")
    p_export.add_argument("--output", type=Path, required=True, help="Output .onnx file path")
    p_export.add_argument("--espeak-voice", type=str, default=None, help="Override espeak voice for export fallback config")

    # synth-test
    p_synth = sub.add_parser("synth-test", help="Synthesize a test WAV from an exported model")
    p_synth.add_argument("--model", type=Path, required=True, help="Path to .onnx file or directory containing .onnx and .json")
    p_synth.add_argument("--text", type=str, required=True, help="Text to synthesize")
    p_synth.add_argument("--out-file", type=Path, default=Path("test_synth.wav"), help="Output WAV file")

    # checkpoint synth
    p_synth_ckpt = sub.add_parser("synth-checkpoint", help="Synthesize a test WAV directly from a checkpoint using VITS interpreter")
    p_synth_ckpt.add_argument("--checkpoint", type=Path, required=True, help="Path to .ckpt checkpoint")
    p_synth_ckpt.add_argument("--text", type=str, required=True, help="Text to synthesize")
    p_synth_ckpt.add_argument("--out-file", type=Path, default=Path("test_synth_from_ckpt.wav"), help="Output WAV file")
    p_synth_ckpt.add_argument("--espeak-voice", type=str, default="en-us", help="espeak voice for phonemization")

    args = parser.parse_args()

    if args.cmd in ("init", "prepare"):
        try:
            prepare_dataset(
                args.samples_dir,
                args.out_dir,
                args.lang,
                args.quality,
                no_convert=args.no_convert,
                config_name=args.voice_name,
            )
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"ffmpeg failed. Make sure ffmpeg is installed: {e}")

    elif args.cmd == "fetch-base":
        ckpt = fetch_base_checkpoint(args.dest_dir, quality=args.quality, model_id=(args.model_id or None))
        print("Base checkpoint available at:", ckpt)

    elif args.cmd == "train":
        out_dir = args.out_dir
        ckpt_path = args.ckpt
        if not ckpt_path:
            # attempt to fetch into default location
            default_dest = Path.home() / ".piper" / "checkpoints" / args.quality
            ckpt_path = fetch_base_checkpoint(default_dest, quality=args.quality)
        run_train(
            out_dir,
            ckpt_path,
            args.voice_name,
            args.quality,
            args.gpu,
            args.run,
            args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            espeak_voice=args.espeak_voice,
        )

    elif args.cmd == "synth-checkpoint":
        synth_test_checkpoint(args.checkpoint, args.text, args.out_file, espeak_voice=args.espeak_voice)

    elif args.cmd == "export":
        export_onnx(args.checkpoint, args.output, espeak_voice=args.espeak_voice)

    elif args.cmd == "synth-test":
        synth_test(args.model, args.text, args.out_file)


if __name__ == "__main__":
    main()
