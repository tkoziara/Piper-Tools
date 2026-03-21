#!/usr/bin/env bash
set -euo pipefail

# Piper-Tools/bootstrap.sh
# Full setup from empty folder: system deps, clone, venv, libs.
# Run from within ./Piper-Tools:  cd Piper-Tools && ./bootstrap.sh

# 1) System dependencies (Debian/Ubuntu; adjust for other distros)
if ! command -v apt >/dev/null 2>&1; then
  echo "apt not found; please install build-essential cmake ninja-build python3-venv python3-dev pkg-config ffmpeg libsndfile1 git curl manually."
else
  echo "Installing system dependencies..."
  sudo apt update
  sudo apt install -y build-essential cmake ninja-build python3-venv python3-dev pkg-config ffmpeg libsndfile1 git curl
fi

# 2) Clone piper1-gpl if it doesn't exist
if [ ! -d "piper1-gpl" ]; then
  echo "Cloning piper1-gpl source..."
  git clone https://github.com/rhasspy/piper1-gpl.git piper1-gpl
else
  echo "piper1-gpl directory already exists, skipping clone (or update with git -C piper1-gpl pull)."
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
python3 -m pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu || true
python3 -m pip install --upgrade piper-tts onnxscript flask openai-whisper soundfile

# 5) Quick sanity print
echo "Bootstrap complete."
echo "activate with: source .venv/bin/activate"

echo "To train sample data, run: python train.py init --samples-dir ../samples --out-dir training_pl --lang pl"
echo "Then: ./train_and_export.sh --out-dir training_pl --voice-name tomek_pl --quality medium --yes --epochs 3 --rounds 3"
