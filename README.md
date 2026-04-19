# Piper TTS local and Google Collab runners

The training/export wrapper (`train_and_export.sh`) now looks in the dataset
folder for previous checkpoints and will use them automatically; if none are
found it falls back to either the language‑specific reference snapshot or the
user’s `~/.piper/checkpoints` directory (see TRAIN.md for details).

Files added:

- `install_system_deps.sh` — script to install system packages (Debian/Ubuntu).
- `bootstrap.sh` — full environment bootstrap: installs system packages, clones `piper1-gpl` if missing, creates/activates a Python virtualenv, installs and pins Python packages (optionally using a CPU-pinned torch wheel to improve ONNX export reliability), builds native extensions (espeak bridge), and performs an editable install of the Piper source. Use `./bootstrap.sh --skip_sys_deps` to skip OS package installation or `--delete` to remove the checkout and venv.
- `synth.py` — runtime script that synthesizes either built‑in sample texts or a specified ONNX model into WAV files via `piper`. It accepts a `--model` path for testing intermediate checkpoints.
 - `train.py` — training and export CLI wrapper used to run training, perform exports to ONNX, and run synth-tests; supports checkpoint handling and resume.
- `synth.sh` — wrapper to activate `.venv` and run `synth.py`.
- `sample_en.txt` — English input text.
- `sample_pl.txt` — Polish input text.
- `Training_{EN,PL}.ipynb` — notebooks for training in Google Colab inspired by
  https://github.com/natlamir/ProjectFiles/blob/main/Piper/Piper_Training.ipynb
- `Training.md` — a training parameters tuning guide for `Training.ipynb` notebook

Usage (after you install system deps):

> **Note**: when preparing a training dataset with `python3 train.py init`,
> the generated `metadata.csv` uses bare file names (e.g. `0001.wav`) rather
> than `wavs/0001.wav`.  The training code will prepend the path to the
> `wavs/` directory automatically, so avoid duplicating that prefix in your
> CSV.


You can manually create a virtual environment as shown below, or simply run
`./setup_venv.sh` which takes care of all Python dependencies and training
extras (see also **reset** option).

```bash
# create/activate environment and install runtime packages
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install piper-tts
```

# training helper script
You can run the `train_and_export.sh` wrapper to take a prepared dataset,
fetch a base checkpoint, and perform training/export/synth-test in one step.
It accepts the usual `--out-dir`, `--voice-name`, `--quality` options and also
`--epochs` for quick trials (e.g. `--epochs 3`) along with `--yes` to execute.
A few extra flags are available for convenience:

* `--rounds N` – perform N sequential rounds (exports/synth-tests are created
  after each).  Rerunning with a larger total causes the script to resume
  rather than overwrite previous results.
* `--batch-size N` – override the batch size used for training (defaults to a
  value derived from the `--quality` preset: 32/16/8 for low/medium/high).
* `--text`      – override the sample sentence used for synth-test; if omitted
  a default English or Polish sentence is chosen automatically based on the
  dataset language.
* `--attempts M` – if the trainer subprocess is killed by the OS (e.g. OOM),
  retry the same round up to M times (default 3).  A manual Ctrl+C still
  terminates the wrapper immediately.
The script also supports a `--rounds` argument to perform multiple sequential
training rounds; if you rerun the wrapper with a larger total round count it
will detect existing `*-r<N>.onnx` outputs and resume numbering instead of
overwriting them, continuing from the latest checkpoint.

   By default the script looks for a file named
   `~/.piper/checkpoints/<quality>/base_checkpoint.ckpt`.  This file is just
   whatever checkpoint you or the `fetch-base` command most recently downloaded
   into that folder.  If you want to start from a different checkpoint you can
   either delete/replace that file or simply invoke the wrapper with
   `--ckpt /path/to/desired.ckpt`.  Likewise, `python3 train.py fetch-base`
   supports `--model-id`/`--model-pattern` to pick a specific snapshot file
   from the Hugging Face dataset.

   The export step requires the `onnxscript` package; the setup script installs
   it automatically (`pip install onnxscript`). If you run into an export error
   you can manually add that dependency.

   **Caveat:** exporting Piper’s VITS generator to ONNX is tricky and may
   fail due to current PyTorch exporter limitations (symbolic‑shape errors,
   guard violations in spline code, etc.).  If export fails the wrapper will
   print a note – you can either try a different PyTorch version, debug with
   `--debug`, or simply skip export and use the trained checkpoint directly.

```bash
# download voices
mkdir -p voices
python3 -m piper.download_voices --data-dir voices en_US-lessac-medium pl_PL-gosia-medium

# alternatively:
# ./setup_venv.sh          # first-time setup
# ./setup_venv.sh --reset  # recreate environment from scratch

# basic run (built-in voices)
python3 synth.py

# synthesize and play the result instead of writing to disk
python3 synth.py --voice en_US-lessac-medium --text "hello" --play

# specify a model file produced by training/export
python3 synth.py --model rounds/round1/model.onnx --text "check one two" --out-file r1.wav

# the same but play immediately (no file created by default):
python3 synth.py --model rounds/round1/model.onnx --text "check one two" --play
```
Outputs: `out_en.wav` and `out_pl.wav` in repository root.

# sample collector web tool
You can build a small dataset interactively by running the bundled
`generate_samples.py` script. It uses a Whisper model to transcribe your
microphone input and lets you correct the text before saving.

```
python generate_samples.py --lang en samples/en    # english samples go here
python generate_samples.py --lang pl samples/pl    # polish samples (ProbkaN)
```

The server listens on http://localhost:8765; open that URL in a browser,
record a short clip, edit the recognised sentence and click **Save**.  The
corresponding `.wav`/`.txt` pair will be appended to the target directory.

(Requires `flask` and `whisper` in the venv; `setup_venv.sh` now installs them.)
