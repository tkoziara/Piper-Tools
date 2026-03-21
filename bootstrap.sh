#!/usr/bin/env bash
set -euo pipefail

# Piper-Tools/bootstrap.sh
# Full setup from empty folder: system deps, clone, venv, libs.
# Run from within ./Piper-Tools:  cd Piper-Tools && ./bootstrap.sh
# Use --delete to remove cloned checkout and venv:
#   ./bootstrap.sh --delete

CLEANUP="false"
SKIP_SYS_DEPS="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --delete)
      CLEANUP="true"
      shift
      ;;
    --skip_sys_deps)
      SKIP_SYS_DEPS="true"
      shift
      ;;
    *)
      echo "Unknown flag: $1"
      exit 1
      ;;
  esac
done

if [ "$CLEANUP" = "true" ]; then
  echo "Deleting piper1-gpl and .venv..."
  python3 - <<'PY'
import pathlib, shutil
for p in [pathlib.Path('piper1-gpl'), pathlib.Path('.venv')]:
    if p.is_symlink() or p.exists():
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
PY
  echo "Deleted. Exiting."
  exit 0
fi

# 1) System dependencies (Debian/Ubuntu; adjust for other distros)
if [ "$SKIP_SYS_DEPS" = "false" ]; then
  if ! command -v apt >/dev/null 2>&1; then
  echo "apt not found; please install build-essential cmake ninja-build python3-venv python3-dev pkg-config ffmpeg libsndfile1 git curl manually."
  else
    echo "Installing system dependencies..."
    sudo apt update
    sudo apt install -y build-essential cmake ninja-build python3-venv python3-dev pkg-config ffmpeg libsndfile1 git curl
  fi
fi

# 2) Setup piper1-gpl source path (prefer existing working checkout at ~/Piper/piper1-gpl)
EXISTING_PIPER_PATH="${HOME}/Piper/piper1-gpl"
PIPER_SRC_PATH="piper1-gpl"
if [ -d "$EXISTING_PIPER_PATH" ]; then
  echo "Found existing working checkout at $EXISTING_PIPER_PATH. Reusing it as source."
  PIPER_SRC_PATH="$EXISTING_PIPER_PATH"
  if [ ! -d "piper1-gpl" ]; then
    ln -s "$EXISTING_PIPER_PATH" piper1-gpl
  fi
fi

if [ ! -d "piper1-gpl" ] && [ "$PIPER_SRC_PATH" = "piper1-gpl" ]; then
  echo "Cloning piper1-gpl source..."
  if git ls-remote git@github.com:OHF-Voice/piper1-gpl.git >/dev/null 2>&1; then
    git clone git@github.com:OHF-Voice/piper1-gpl.git piper1-gpl
  else
    echo "SSH clone failed or SSH key not configured; falling back to HTTPS (may prompt for credentials)."
    git clone https://github.com/OHF-Voice/piper1-gpl.git piper1-gpl
  fi
else
  echo "piper1-gpl source already available at $PIPER_SRC_PATH."
fi

# 3) Set up Python virtualenv
if [ ! -d ".venv" ]; then
  echo "Creating virtualenv .venv..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python3 -m pip install --upgrade pip

# 4) Install required Python packages
echo "Installing Python dependencies..."
# Pin torch to a known stable CPU version for Piper so ONNX export is more reliable
PYTORCH_VERSION="2.2.2"
python3 -m pip install --upgrade "torch==${PYTORCH_VERSION}+cpu" "torchvision==0.17.2+cpu" "torchaudio==2.2.2+cpu" --extra-index-url https://download.pytorch.org/whl/cpu || \
python3 -m pip install --upgrade "torch==${PYTORCH_VERSION}" "torchvision==0.17.2" "torchaudio==2.2.2"
# Enforce a numpy version compatible with existing torch/compiled extensions.
python3 -m pip install --upgrade "numpy<2"
# Install direct dependencies needed for training pipeline.
python3 -m pip install --upgrade piper-tts onnxscript flask openai-whisper soundfile librosa lightning pytorch-lightning pysilero-vad pathvalidate jsonargparse[signatures] || true

if [ -d "$PIPER_SRC_PATH" ]; then
  echo "Installing Piper source from: $PIPER_SRC_PATH"
  python3 -m pip install --upgrade -e "$PIPER_SRC_PATH[train]"
  if [ -f "$PIPER_SRC_PATH/src/piper/espeakbridge.so" ]; then
    echo "espeakbridge native module already built at $PIPER_SRC_PATH/src/piper/espeakbridge.so"
  fi
  # Sanity check that the espeak native bridge is importable.
  python3 - <<'PY'
import importlib.util, sys
spec = importlib.util.find_spec('piper.espeakbridge')
if spec is None:
    sys.exit('ERROR: piper.espeakbridge extension module not found after install.')
print('INFO: piper.espeakbridge module is available at', spec.origin)
PY
else
  echo "ERROR: piper1-gpl source path $PIPER_SRC_PATH not found"
  exit 1
fi

# 5) Quick sanity print
echo "Bootstrap complete."
echo "activate with: source .venv/bin/activate"

echo "To train sample data, run: python train.py init --samples-dir ../samples --out-dir training_pl --lang pl"
echo "Then: ./train_and_export.sh --out-dir training_pl --voice-name tomek_pl --quality medium --yes --epochs 3 --rounds 3"
