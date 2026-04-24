# Piper TTS local and Google Colab runners

This repository contains scripts and notebooks for preparing Piper TTS datasets,
training models, exporting them to ONNX, and running synthesis tests.

A common workflow is:
1. prepare or collect data locally with `record_samples.py` and set them up for training with `train.py init`
2. train a model in Google Colab using `Training_EN.ipynb` / `Training_PL.ipynb`
3. export the trained checkpoint with `train.py export` or `train_and_export.sh`
4. synthesize using `synth.py` or `synth.sh`

Files in this repository:

- `install_system_deps.sh` — install required Debian/Ubuntu system packages.
- `bootstrap.sh` — bootstrap the repo and Python environment; optionally installs system packages, clones `piper1-gpl`, builds native extensions, and installs Python dependencies into a venv.
- `setup_venv.sh` — create and activate the local `.venv`, install Python requirements, and prepare the training/export environment.
- `train.py` — main CLI wrapper for dataset initialization, training, checkpoint handling, export, and synth-testing.
- `train_and_export.sh` — convenience wrapper that can prepare a dataset, download a base checkpoint, train, export to ONNX, and run synthesis tests in a single sequence.
- `synth.py` — runtime synthesis script for built-in voices or a trained ONNX model.
- `synth.sh` — wrapper to activate `.venv` and run `synth.py`.
- `checkpoint.py` — list and download Hugging Face Piper checkpoints.
- `record_samples.py` — interactive web tool for collecting microphone samples plus transcriptions.
- `sample_en.txt` — English test sentence.
- `sample_pl.txt` — Polish test sentence.
- `Training_EN.ipynb` — Google Colab notebook for English training.
- `Training_PL.ipynb` — Google Colab notebook for Polish training.
- `Training.md` — tuning guide for notebook training parameters and phase presets.
- `README.md` — this file.

> **Note:** `python3 train.py init` generates `metadata.csv` with bare audio file names
> (for example `0001.wav`). Do not prefix them with `wavs/`; the training loader
> prepends the audio directory automatically.

## Quick start

Install system dependencies first:

```bash
./install_system_deps.sh
```

Create and activate the Python environment, or use the helper script:

```bash
./setup_venv.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install piper-tts
```

## Training and export

The simplest training helper is `train_and_export.sh`. It can:

- prepare a dataset and fetch a base checkpoint
- run training
- export the trained model to ONNX
- perform a synth test after export

Key flags:

- `--out-dir`, `--voice-name`, `--quality`
- `--epochs` — quick trial mode
- `--rounds N` — perform N sequential export rounds and resume numbering across runs
- `--batch-size N` — override the default batch size
- `--text` — override the synth-test sentence
- `--attempts M` — retry a failed training round up to M times
- `--ckpt /path/to/ckpt` — start from a specific checkpoint

By default `train_and_export.sh` looks for prior checkpoints in the dataset folder
and will reuse them automatically. If none are found it falls back to a language-
specific reference snapshot or `~/.piper/checkpoints`.

The export step requires `onnxscript`, which `setup_venv.sh` installs. If ONNX
export fails, you can still use the trained checkpoint directly or try a different
PyTorch version.

## Basic synthesis

```bash
python3 synth.py
python3 synth.py --voice en_US-lessac-medium --text "hello" --play
python3 synth.py --model rounds/round1/model.onnx --text "check one two" --out-file r1.wav
python3 synth.py --model rounds/round1/model.onnx --text "check one two" --play
```

By default the output files are written to the repository root as `out_en.wav`
and `out_pl.wav`.

## Sample collection

Use `record_samples.py` to collect `.wav`/`.txt` pairs in a browser:

```bash
python record_samples.py --lang en samples/en
python record_samples.py --lang pl samples/pl
```

Then open `http://localhost:8765` and record a short clip. You can edit the
recognized sentence before saving. The script writes the pair into the
chosen sample directory.

This requires `flask` and `whisper` in the venv; `setup_venv.sh` installs them.