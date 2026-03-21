#!/usr/bin/env bash
set -euo pipefail
echo "This script installs system packages required to build/run Piper (Debian/Ubuntu)."
echo "Run: bash install_system_deps.sh"

sudo apt update
sudo apt install -y build-essential cmake ninja-build python3-venv python3-dev pkg-config ffmpeg libsndfile1 git curl

echo "System packages installed. If you don't have sudo access, install the listed packages manually."
