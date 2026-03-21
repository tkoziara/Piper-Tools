#!/usr/bin/env bash
set -euo pipefail

# if the user hits Ctrl+C, bail out immediately rather than retrying
trap 'echo "Interrupted by user"; exit 130' INT

# train_and_export.sh
# Wrapper to run a dry-run of training, optionally execute training, export ONNX, and run a synth-test.

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

# Suppress noisy FutureWarning messages (e.g. from torch.load in downstream libs)
# This prevents spurious warnings from cluttering stdout/stderr during training.
export PYTHONWARNINGS="ignore::FutureWarning"

usage() {
  cat <<EOF
Usage: $0 --out-dir OUT [--ckpt CKPT] --voice-name NAME [--quality q] [--batch-size N] [--text TEXT] [--rounds N] [--attempts M] [--yes]

Options:
  --out-dir    Dataset folder prepared by train.py init
  --ckpt       Base checkpoint path (optional, will fetch if omitted)
  --voice-name Voice name to use for training
  --quality    low|medium|high (default: medium)
  --batch-size INT  Training batch size to pass to Piper (default: preset value)
  --epochs     Override number of epochs for testing (default from quality)
  --rounds     Total number of sequential training rounds to perform
                (default: 1).  If output files from earlier rounds already
                exist, the script will resume from the next index and only
                execute the remaining rounds; previously generated ONNX
                / wav files are left intact.
  --attempts   Number of attempts to run each round if the trainer is
                killed by the OS (default: 3).  Ctrl+C still stops the
                wrapper immediately.
  --text       Test text for synth-test (defaults to an English or Polish
                sentence depending on language inferred from the dataset or
                checkpoint.)
  --yes        Actually run training (otherwise dry-run only)
EOF
}

OUT_DIR=""
CKPT=""
VOICE_NAME="voice"
QUALITY="medium"
BATCH_SIZE=""          # default determined later from quality presets
TEXT_EN="This is my perfect cloned voice and I love you for it."
TEXT_PL="To jest mój idealny sklonowany głos i kocham cię za to."
EPOCHS=""
ROUNDS=1
ATTEMPTS=3
DO_RUN="false"
CPUS=10

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2;;
    --ckpt) CKPT="$2"; shift 2;;
    --voice-name) VOICE_NAME="$2"; shift 2;;
    --quality) QUALITY="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --text) TEXT="$2"; shift 2;;
    --rounds) ROUNDS="$2"; shift 2;;
    --attempts) ATTEMPTS="$2"; shift 2;;
    --yes) DO_RUN="true"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [ -z "$OUT_DIR" ]; then
  echo "--out-dir is required"; usage; exit 1
fi

# figure out language of dataset (english vs polish) so we can pick sensible
# default test text if the user didn't supply one.  we look at voice_config.json
# if present, otherwise guess from the out-dir name.
lang_guess="en"
# prefer a dataset config backup if it exists; training may later
# overwrite voice_config.json with a model-derived file.
cfg_path="$OUT_DIR/voice_config.dataset.json"
if [ ! -f "$cfg_path" ]; then
  cfg_path="$OUT_DIR/voice_config.json"
fi
if [ -f "$cfg_path" ]; then
  esv=$(python3 - <<PY
import json
try:
    j=json.load(open('$cfg_path'))
    print(j.get('data',{}).get('espeak_voice',''))
except Exception:
    pass
PY
  )
  if echo "$esv" | grep -qi "pl"; then
    lang_guess="pl"
  fi
else
  if echo "$OUT_DIR" | grep -qi "pl"; then
    lang_guess="pl"
  fi
fi

# set TEXT default based on language if not provided explicitly
if [ -z "${TEXT-}" ]; then
  if [ "$lang_guess" = "pl" ]; then
    TEXT="$TEXT_PL"
  else
    TEXT="$TEXT_EN"
  fi
fi

# derive an espeak voice string we will forward to Piper (override later if
# user explicitly passes --text or other options)
if [ "$lang_guess" = "pl" ]; then
  ESPEAK_VOICE="pl"
else
  ESPEAK_VOICE="en-us"
fi

# If CKPT not provided, try to auto-detect from OUT_DIR/voice_config.json or from ~/.piper/checkpoints
# if the caller didn't specify a checkpoint, first see if the dataset
# folder already contains one from a previous run.  this lets us continue
# training without forcing the user to re-specify --ckpt.
if [ -z "$CKPT" ]; then
  if [ -d "$OUT_DIR/tts_output" ]; then
    CKPT_PREV=$(find "$OUT_DIR/tts_output" -type f -name "*.ckpt" -print0 \
        | xargs -0 ls -t 2>/dev/null | head -n1 || true)
    if [ -n "$CKPT_PREV" ]; then
      CKPT="$CKPT_PREV"
      echo "Using existing checkpoint from output directory: $CKPT"
    fi
  fi
fi

if [ -z "$CKPT" ]; then
  cfg="$OUT_DIR/voice_config.dataset.json"
  lang_guess="en"
  if [ ! -f "$cfg" ]; then
    cfg="$OUT_DIR/voice_config.json"
  fi
  if [ -f "$cfg" ]; then
    # try to read espeak_voice from config
    esv=$(python3 - <<PY
import json,sys
try:
    j=json.load(open('$cfg'))
    v=j.get('data',{}).get('espeak_voice','')
    print(v)
except Exception:
    sys.exit(0)
PY
)
    if echo "$esv" | grep -qi "pl"; then
      lang_guess="pl"
    fi
  else
    # fallback: infer from out dir name
    if echo "$OUT_DIR" | grep -qi "pl"; then
      lang_guess="pl"
    fi
  fi

  echo "No --ckpt provided; attempting to find a default checkpoint for language: $lang_guess (quality: $QUALITY) under ~/.piper/checkpoints"
  # search for ckpt files under ~/.piper/checkpoints/<quality>
  base_ckpt_dir="$HOME/.piper/checkpoints/$QUALITY"
  CKPT_CAND=""
  if [ -d "$base_ckpt_dir" ]; then
    # Prefer files whose path contains the language code (pl or en).  we do
    # **not** fall back to a random checkpoint from the quality folder;
    # otherwise a blob file (which has no language tag) may be chosen.
    if [ "$lang_guess" = "pl" ]; then
      # look for a path segment containing "pl/" to avoid false positives
      CKPT_CAND=$(find "$base_ckpt_dir" -type f -name "*.ckpt" -ipath "*/pl/*" | head -n1 || true)
    else
      # English may appear as en/, en_US etc.
      CKPT_CAND=$(find "$base_ckpt_dir" -type f -name "*.ckpt" -ipath "*/en/*" | head -n1 || true)
    fi
  fi

  # if still empty, try global search under ~/.piper/checkpoints
  if [ -z "$CKPT_CAND" ]; then
    if [ "$lang_guess" = "pl" ]; then
      CKPT_CAND=$(find "$HOME/.piper/checkpoints" -type f -name "*.ckpt" -ipath "*/pl/*" | head -n1 || true)
    else
      CKPT_CAND=$(find "$HOME/.piper/checkpoints" -type f -name "*.ckpt" -ipath "*/en/*" | head -n1 || true)
    fi
  fi

  # fallback to base_checkpoint.ckpt
  if [ -z "$CKPT_CAND" ] && [ -f "$HOME/.piper/checkpoints/$QUALITY/base_checkpoint.ckpt" ]; then
    CKPT_CAND="$HOME/.piper/checkpoints/$QUALITY/base_checkpoint.ckpt"
  fi

  if [ -n "$CKPT_CAND" ]; then
    CKPT="$CKPT_CAND"
    echo "Auto-selected checkpoint: $CKPT"
  else
    # no candidates found – fall back to explicit hard-coded defaults per
    # language.  these are large snapshots we know exist in the HF dataset.
    if [ "$lang_guess" = "pl" ]; then
      CKPT="$HOME/.piper/checkpoints/medium/datasets--rhasspy--piper-checkpoints/snapshots/52588227e5a29f8c2afc6c31280e42119760ac86/pl/pl_PL/darkman/medium/epoch=4909-step=1454360.ckpt"
      echo "No checkpoint found automatically; defaulting to Polish reference: $CKPT"
    else
      CKPT="$HOME/.piper/checkpoints/medium/datasets--rhasspy--piper-checkpoints/snapshots/52588227e5a29f8c2afc6c31280e42119760ac86/en/en_GB/northern_english_male/medium/epoch=9029-step=2261720.ckpt"
      echo "No checkpoint found automatically; defaulting to English reference: $CKPT"
    fi
  fi
fi

# determine how many rounds we already have output for and adjust

# If the user requested multiple rounds but did not override --epochs,
# avoid running the full preset epochs for each round (which would cause
# excessive total training and likely model degradation).  Instead compute
# a per-round epoch count by dividing the preset total across rounds.
if [ -z "${EPOCHS-}" ] && [ "$ROUNDS" -gt 1 ]; then
  total_epochs=$(python3 - <<PY
import sys
from pathlib import Path
sys.path.insert(0, str(Path("$ROOT").resolve()))
from train import quality_presets
try:
    print(int(quality_presets("$QUALITY")["epochs"]))
except Exception:
    sys.exit(1)
PY
  ) || total_epochs=""
  if [ -n "$total_epochs" ]; then
    per_round=$(( (total_epochs + ROUNDS - 1) / ROUNDS ))
    echo "No --epochs specified and --rounds > 1: running $per_round epochs per round (approx total $total_epochs)"
    EPOCHS="$per_round"
  fi
fi

# establish default batch size if not specified
if [ -z "$BATCH_SIZE" ]; then
  case "$QUALITY" in
    low) BATCH_SIZE=32;;
    high) BATCH_SIZE=8;;
    *) BATCH_SIZE=16;;
  esac
  echo "Using batch size $BATCH_SIZE (quality preset $QUALITY)"
else
  echo "Using custom batch size $BATCH_SIZE"
fi
existing_rounds=0
if [ -d "$OUT_DIR" ]; then
  # look for files matching the pattern and extract highest index
  existing_rounds=$(find "$OUT_DIR" -maxdepth 1 -type f -name "${VOICE_NAME}-${QUALITY}-r*.onnx" \
    | sed -n 's/.*-r\([0-9]\+\)\.onnx$/\1/p' \
    | sort -n | tail -n1 || true)
  existing_rounds=${existing_rounds:-0}
fi
if [ "$existing_rounds" -ge "$ROUNDS" ]; then
  echo "Detected $existing_rounds existing rounds (>= requested $ROUNDS); nothing to do."
  exit 0
fi
start_round=$((existing_rounds + 1))
remaining=$((ROUNDS - existing_rounds))
if [ "$DO_RUN" != "true" ]; then
  echo "Dry-run: printing training + export/synth commands"
  echo "Will execute rounds $start_round through $ROUNDS (existing $existing_rounds), up to $ATTEMPTS attempts per round if the trainer is killed."
  for round in $(seq $start_round $ROUNDS); do
    echo "# round $round/$ROUNDS (using checkpoint: $CKPT)"
    # simply print the command we would run; avoid invoking train.py so we don't
    # perform sanitization or other side effects during dry-run.
    printf "python3 %s/train.py train --out-dir %s --ckpt %s --voice-name %s --quality %s --batch-size %s --num-workers %s --espeak-voice %s" "$ROOT" "$OUT_DIR" "$CKPT" "$VOICE_NAME" "$QUALITY" "$BATCH_SIZE" "$CPUS" "$ESPEAK_VOICE"
    if [ -n "$EPOCHS" ]; then
      printf " --epochs %s" "$EPOCHS"
    fi
    echo

    echo "# export for round $round"
    echo python3 "$ROOT/train.py" export --checkpoint '<LATEST_CKPT_FROM_OUTDIR>' --output "$OUT_DIR/${VOICE_NAME}-${QUALITY}-r${round}.onnx"

    echo "# synth-test for round $round"
    echo python3 "$ROOT/train.py" synth-test --model \"$OUT_DIR/${VOICE_NAME}-${QUALITY}-r${round}.onnx\" --text \"$TEXT\" --out-file \"$OUT_DIR/${VOICE_NAME}-${QUALITY}-r${round}.wav\"
  done
else
  # running for real; just summarise what will happen
  echo "Will execute rounds $start_round through $ROUNDS (existing $existing_rounds), up to $ATTEMPTS attempts per round if the trainer is killed."
fi

if [ "$DO_RUN" = "true" ]; then
  echo "Running training now (CPU mode)..."

  # Ensure CPU PyTorch is available in the venv; install CPU-only wheel if missing
  if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "PyTorch not found in venv; installing CPU-only PyTorch (may take some minutes)..."
    python3 -m pip install --upgrade pip
    # Use PyTorch CPU wheels index
    python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
  fi

  # Limit number of threads for CPU training
  export OMP_NUM_THREADS=$CPUS
  export MKL_NUM_THREADS=$CPUS
  export OPENBLAS_NUM_THREADS=$CPUS
  export PYTORCH_NUM_THREADS=$CPUS

  prev_ckpt="$CKPT"
  echo "Starting from round $start_round (found $existing_rounds existing)"
  for round in $(seq $start_round $ROUNDS); do
    echo "=== round $round/$ROUNDS ==="
    attempt=1
    while true; do
      echo "  attempt $attempt/$ATTEMPTS"
      python3 "$ROOT/train.py" train --out-dir "$OUT_DIR" --ckpt "$prev_ckpt" --voice-name "$VOICE_NAME" --quality "$QUALITY" --batch-size "$BATCH_SIZE" --num-workers "$CPUS" --espeak-voice "$ESPEAK_VOICE" $( [ -n "$EPOCHS" ] && printf -- "--epochs %s" "$EPOCHS" ) --run
      rc=$?
      if [ $rc -eq 0 ]; then
        break
      fi
      # check for kill-by-signal (exit code 128+sig)
      if [ $rc -ge 128 ]; then
        sig=$((rc - 128))
      else
        sig=0
      fi
      if [ $sig -eq 9 ]; then
        echo "  training process was killed by SIGKILL (rc=$rc)"
        if [ $attempt -lt $ATTEMPTS ]; then
          attempt=$((attempt + 1))
          echo "  retrying round $round (next attempt $attempt/$ATTEMPTS)..."
          continue
        else
          echo "Round $round failed after $ATTEMPTS attempts; aborting."
          exit 2
        fi
      fi
      echo "Training failed on round $round with exit code $rc; aborting."
      exit 2
    done

    # update checkpoint for next round
    if [ -d "$OUT_DIR/tts_output" ]; then
      prev_ckpt=$(find "$OUT_DIR/tts_output" -type f -name "*.ckpt" -print0 \
        | xargs -0 ls -t 2>/dev/null | head -n1 || true)
      CKPT="$prev_ckpt"
    fi

    # export and synth-test for this round
    OUT_ONNX="$OUT_DIR/${VOICE_NAME}-${QUALITY}-r${round}.onnx"
    echo "Exporting checkpoint $prev_ckpt -> $OUT_ONNX"
    if ! python3 "$ROOT/train.py" export --checkpoint "$prev_ckpt" --output "$OUT_ONNX"; then
      echo "Export failed for round $round" >&2
    fi
    # Do not copy the dataset `voice_config.json` into the model `.json`.
    # The exporter should create a compatible model config; copying the
    # dataset config produces an invalid model config (missing model keys).
    echo "Synth-testing round $round"
    python3 "$ROOT/train.py" synth-test --model "$OUT_ONNX" --text "$TEXT" --out-file "$OUT_DIR/${VOICE_NAME}-${QUALITY}-r${round}.wav" || true
  done
else
  echo "Not running training. Rerun with --yes to execute."
  exit 0
fi

# After training, locate a checkpoint under OUT_DIR/tts_output
CKPT_DIR="$OUT_DIR/tts_output"
if [ -d "$CKPT_DIR" ]; then
  # search recursively for any .ckpt files and take the newest by mtime
  LATEST_CKPT=$(find "$CKPT_DIR" -type f -name "*.ckpt" -print0 \
    | xargs -0 ls -t 2>/dev/null | head -n1 || true)
  if [ -z "$LATEST_CKPT" ]; then
    echo "No checkpoint found under $CKPT_DIR"
    exit 0
  fi
else
  echo "No tts_output folder found at $CKPT_DIR"
  exit 0
fi

OUT_ONNX="$OUT_DIR/${VOICE_NAME}-${QUALITY}.onnx"
echo "Exporting checkpoint $LATEST_CKPT -> $OUT_ONNX"
if ! python3 "$ROOT/train.py" export --checkpoint "$LATEST_CKPT" --output "$OUT_ONNX"; then
  echo "Export failed.  This usually means a dependency is missing (e.g."
  echo "  ModuleNotFoundError: No module named 'onnxscript')."
  echo "Install 'onnxscript' in the venv or re-run ./setup_venv.sh to add it."
  exit 0
fi

# make sure an associated .json config exists for runtime
if [ ! -f "$OUT_ONNX.json" ]; then
  echo "Warning: exporter did not produce a model .json for $OUT_ONNX; not copying dataset voice_config.json (would be invalid)." >&2
fi

echo "Running synth-test with exported model"
python3 "$ROOT/train.py" synth-test --model "$OUT_ONNX" --text "$TEXT" --out-file "$OUT_DIR/test_synth.wav"

echo "Done. test_synth.wav written to $OUT_DIR"
