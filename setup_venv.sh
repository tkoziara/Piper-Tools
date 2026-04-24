#!/usr/bin/env bash
set -euo pipefail

# setup_venv.sh
#
# Create or reset the Python virtual environment for Piper TTS development.
#
# This script performs all of the steps necessary to reproduce the working
# environment used in this project, including:
#  * system package hints (see install_system_deps.sh for apt instructions)
#  * creating a Python venv at .venv
#  * installing runtime dependencies (piper-tts, onnxruntime, etc.)
#  * installing CPU-only PyTorch (training uses CPU by default)
#  * cloning the Piper source tree and installing the training extras
#  * building the native C/Cython extensions required for training
#
# Usage:
#   ./setup_venv.sh          # create environment if missing
#   ./setup_venv.sh --reset  # wipe and rebuild from scratch
#
# After the script finishes, you can activate the environment with:
#   source .venv/bin/activate
#

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
PIP=""
PIP_INSTALL=""

# parse args
RESET=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --reset) RESET=true; shift;;
        -h|--help)
            cat <<EOF
Usage: $0 [--reset]

Options:
  --reset   Delete existing venv and rebuild from scratch.
  -h        Show this message.
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# optional system-deps message
if [ "$RESET" = true ] || [ ! -d "$VENV" ]; then
    echo "* You should ensure system packages are installed before running this"
    echo "  (see install_system_deps.sh). Typical packages: build-essential cmake"
    echo "  ninja-build pkg-config python3-dev libsndfile1-dev ffmpeg git "
    echo "  python3-venv.  Run: sudo bash install_system_deps.sh"
    echo
fi

if [ "$RESET" = true ] && [ -d "$VENV" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV"
fi

if [ ! -d "$VENV" ]; then
    echo "Creating Python virtual environment at $VENV"
    python3 -m venv "$VENV"
fi

# shellcheck source=/dev/null
. "$VENV/bin/activate"

PIP="$(command -v pip)"
PIP_INSTALL="$PIP install --no-cache-dir --upgrade"

# upgrade packaging tools
$PIP_INSTALL pip setuptools wheel

# install runtime packages
# include onnxscript which PyTorch's ONNX exporter may require
# note: the PyPI package "whisper" is unrelated (RRDtool), so uninstall
# it first to avoid conflicts and then install OpenAI's model.
$PIP_INSTALL piper-tts onnxruntime onnx onnxscript flask deepfilternet
# safe-removal of wrong whisper
$PIP uninstall -y whisper || true
$PIP_INSTALL "openai-whisper"
$PIP_INSTALL "huggingface_hub"

# install or upgrade CPU PyTorch
# PyTorch 2.10+ uses the new `torch.export` ONNX path which is still
# fragile for the VITS model.  For more reliable exports, we default to
# an earlier 2.5.x wheel.  You can override by setting TORCH_VERSION in the
# environment (e.g. "TORCH_VERSION=2.10.0+cpu ./setup_venv.sh").
TORCH_VERSION="${TORCH_VERSION:-2.5.1+cpu}"
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "Installing CPU-only PyTorch ($TORCH_VERSION)..."
    $PIP_INSTALL --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"
else
    # check version, upgrade/downgrade if mismatch
    INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    if [ "$INSTALLED" != "$TORCH_VERSION" ]; then
        echo "Updating PyTorch from $INSTALLED to $TORCH_VERSION"
        $PIP_INSTALL --upgrade --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"
    fi
fi

# clone or update the Piper source tree (needed for training extras & C code)
PIPER_SRC="$ROOT/piper1-gpl"
if [ ! -d "$PIPER_SRC" ]; then
    echo "Cloning Piper repository for training extras..."
    git clone https://github.com/rhasspy/piper "$PIPER_SRC"
else
    echo "Updating Piper source..."
    (cd "$PIPER_SRC" && git pull --ff-only || true)
fi

# install training extras (editable) and build necessary native modules
# some of the build steps require scikit-build & cython
$PIP_INSTALL scikit-build cython

# install training extras (the -e ensures we can rebuild in-place later)
# using --no-deps because dependencies are already installed above or
# will be pulled automatically
python -m pip install -e "$PIPER_SRC[train]"

# compile Cython extensions (monotonic_align etc.)
if [ -f "$PIPER_SRC/build_monotonic_align.sh" ]; then
    echo "Running build_monotonic_align.sh"
    bash "$PIPER_SRC/build_monotonic_align.sh" || true
fi
if [ -f "$PIPER_SRC/setup.py" ]; then
    echo "Compiling extension modules via setup.py"
    (cd "$PIPER_SRC" && python setup.py build_ext --inplace)
fi

# final sanity check
echo "Verifying that piper.train can be imported..."
python - <<'PY'
try:
    import piper.train
    print('IMPORT_OK')
except Exception as e:
    import traceback; traceback.print_exc()
    print('IMPORT_FAIL')
    sys.exit(1)
PY

echo "Virtual environment setup complete. Activate with:\n  source $VENV/bin/activate"
