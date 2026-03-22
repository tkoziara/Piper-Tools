# Training guide — Piper voice cloning (quick examples + detailed options)

This file documents the simplified training workflow implemented by `train.py` in this
workspace. It includes quick usage examples, tunable parameters, tips to obtain smaller
base checkpoints, and practical scenarios for `low`/`medium`/`high` quality runs.

Files you will use

- `train.py` — the runner script (commands: `init`/`prepare`, `fetch-base`, `train`, `export`, `synth-test`).
- `samples/` — your input folder with `en/` and/or `pl/` subfolders containing audio and optional transcripts.
- `training_*` — dataset folders created by `train.py init` containing `wavs/`, `metadata.csv`, and `README_TRAINING.md`.

Quick usage examples

1) Prepare a dataset from your samples (normalize audio to 22050 Hz):

```bash
python3 train.py init --samples-dir samples_en --out-dir training_en --lang en
python3 train.py init --samples-dir samples_pl --out-dir training_pl --lang pl
```

   If you don't have existing audio/text pairs you can build them interactively
   using the web-based helper `generate_samples.py` which uses Whisper to
   transcribe your voice.  See the project README for usage.

2) Fetch a base Piper checkpoint (automatic, uses Hugging Face `rhasspy/piper-checkpoints` by default):

```bash
python3 train.py fetch-base --dest-dir ~/.piper/checkpoints/medium --quality medium
```

   The command will download a checkpoint file and copy it to
   `~/.piper/checkpoints/medium/base_checkpoint.ckpt` (the wrapper and
   training commands always look for a file of this name when `--ckpt` is
   omitted).  That `base_checkpoint.ckpt` is simply a convenience alias – it
   is **not** special metadata from the repository.  You are free to delete or
   replace it at any time.  If it is removed, the next `train` invocation will
   run `fetch-base` again, which may pick the first matching file in the HF
   dataset.

   To choose a particular starting point, either:

   * pass `--ckpt /path/to/your/preferred.ckpt` to `train.py train` or the
     wrapper (you can also give `--batch-size` directly to `train.py train`).
   * run `python3 train.py fetch-base` with `--model-id` or `--model-pattern`
     to select the exact file you want (see the earlier section on targeted
     downloads).

   The wrapper now has a bit more intelligence when you omit `--ckpt`:
   1. if the output directory already contains checkpoints from a previous
      run (`tts_output/.../*.ckpt`), the most recent one is used so you can
      continue training without specifying anything;
   2. otherwise the normal `~/.piper/checkpoints/...` search is performed
      (filtering by language as before);
   3. if that search still yields nothing, the script falls back to a
      hard‑coded reference file specific to the guessed language (the long
      snapshot paths shown earlier).  You can override this by supplying
      `--ckpt` explicitly or by deleting the default files in
      `~/.piper/checkpoints/<quality>`.

   The wrapper also supports a new `--rounds N` option.  Rather than
   asking for a large number of epochs in one go, you can perform the same
   total training in N smaller rounds; after each round the latest checkpoint
   produced in `tts_output` is used as the starting point for the next.  If
   you rerun the wrapper later with a larger total round count, it will
   detect the existing `*-r<k>.onnx` exports and resume numbering – e.g. a
   run that stopped at round 6 and is restarted with `--rounds 10` will only
   execute rounds 7–10, leaving the previous files untouched.  This makes it
   safe to abort/train repeatedly without losing earlier results.
   For convenience the script exports and synthesizes a short test file
   (`test_synth-r<k>.wav`) after every round so you can monitor progress
   without manual intervention.

> **Note:** the training wrapper (`train_and_export.sh`) and `train.py` automatically
> "sanitize" base checkpoints to remove extraneous hyperparameters and training
> state which can confuse the Lightning CLI.  You do **not** need to edit the
> downloaded file yourself; the script will make a cleaned copy
> (`base_checkpoint.ckpt.clean.ckpt`) before starting training.

Notes: this step may download large files. See the "Smaller checkpoints" section below if you want to reduce size.

3) Print (dry-run) the Piper training command for a medium-quality fine-tune:

```bash
python3 train.py train --out-dir training_en --ckpt ~/.piper/checkpoints/medium/base_checkpoint.ckpt --voice-name tomek --quality medium
```

4) Actually run training (explicit):

```bash
python3 train.py train --out-dir training_en --ckpt ~/.piper/checkpoints/medium/base_checkpoint.ckpt --voice-name tomek --quality medium --run
```

   You can also use the helper script `train_and_export.sh` which wraps these
   commands and will automatically export and test the model after training. It
   accepts the same `--quality`/`--epochs` flags and a `--yes` switch to actually
   perform the run, for example:

```bash
./train_and_export.sh --out-dir training_en --voice-name tomek --quality medium --epochs 5 --yes

# you can also limit retries if a round is killed by the OS (default 3):
./train_and_export.sh ... --rounds 10 --attempts 5
```

   (the small `--epochs` override will make the wrapper finish quickly.)

   Training outputs (log files and checkpoints) will now be written under
   the `tts_output` subdirectory of your dataset.  The script achieves this by
   passing `--trainer.default_root_dir training_en/tts_output` to the Piper
   CLI, so Lightning places `lightning_logs/version_x/checkpoints` inside
   that folder.  This matches what `train_and_export.sh` expects and makes it
   easy to locate the generated `.ckpt` files for export.

5) Export the trained checkpoint to ONNX (replace `checkpoint_latest.ckpt` with your real checkpoint):

```bash
python3 train.py export --checkpoint training_en/tts_output/checkpoint_latest.ckpt --output training_en/tomek-medium.onnx
```

   **Note:** recent versions of PyTorch require the `onnxscript` package for
   ONNX export.  If you see an error about `No module named 'onnxscript'`,
   install it via `pip install onnxscript` (the provided `setup_venv.sh` now
   installs it automatically).

   The export itself is fragile with PyTorch 2.6+ because of the new
   `torch.export` pathway; the setup script defaults to installing
   `torch==2.5.1+cpu` which has a much higher success rate.  You can override
   the version by setting the `TORCH_VERSION` environment variable when
   running `setup_venv.sh`, but be prepared for export failures with newer
   releases – see the earlier discussion for workarounds.

   **Warning:** exporting the full VITS model is fragile – PyTorch's
   `torch.onnx.export` (which now uses the `torch.export` pathway) often
   fails with symbolic‑shape or data‑dependent errors inside the spline
   transforms.  This is a limitation of the exporter, not your data.  If
   export trips over a `torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode`
   error or similar, you have a few options:

   1. try an earlier PyTorch release (2.5 or older) where the exporter was
      simpler;
   2. run the export script with `--debug` and/or switch to
      `draft_export()` manually to see more details;
   3. edit `piper1-gpl/src/piper/train/vits/transforms.py` to avoid
      assertions or other non‑exportable constructs (e.g. use
      `guard_or_false`);
   4. skip ONNX entirely and use the model directly via the Piper runtime.

   In short, export may not always succeed; the wrapper will print an
   explanatory message and exit gracefully if it fails.

   The export step uses `torch.onnx` which on newer PyTorch requires the
   `onnxscript` package; if you run into `ModuleNotFoundError: onnxscript`
   simply install it in your environment (`pip install onnxscript`).

6) Test synthesis using the exported model via the helper in `train.py`:

```bash
python3 train.py synth-test --model training_en/tomek-medium.onnx --text "This is my perfect cloned voice and I love you for it." --out-file demo.wav

# default English/Polish text is chosen automatically based on dataset language
# (use --text to override):
python3 train.py synth-test --model training_en/tomek-medium.onnx --out-file demo.wav
```

   Alternatively you can use the standalone runtime script `synth.py` which
   ships with this repository.  After activating the virtual environment run:

```bash
# synthesize using a named voice from voices/ directory
python3 synth.py --voice tomek-medium --text "This is my perfect cloned voice and I love you for it." --out-file demo.wav

# or let the script pick a language-appropriate sample automatically:
python3 synth.py --voice tomek-medium --out-file demo.wav

# or point directly at an exported ONNX file:
python3 synth.py --model training_en/tomek-medium.onnx --text "This is my perfect cloned voice and I love you for it." --out-file demo.wav

# you can also play instead of (or in addition to) saving:
python3 synth.py --voice tomek-medium --play
```

   (the command-line arguments mirror those of `train.py synth-test` when
   using `--model`.)  The `--voice` name corresponds to the ONNX filename
   without extension, and you can use `--voices-dir` to point somewhere
   else.  Use `--list` to see the built‑in sample voices.)

Quality presets — what they change

The `--quality` argument maps to sensible training hyperparameters. Use these presets as starting points:

> **Quick tip:** you can bypass the preset epoch count by supplying
> `--epochs N` to `train.py train`.  This is handy for very fast dry‑run
> tests where you only need a handful of epochs instead of the full 50/200/400.

- `low` — quick experiments and sanity checks
  - epochs: 50
  - batch_size: 32
  - notes: faster, lower-quality output (useful for verifying data pipeline)

- `medium` — default balanced setting
  - epochs: 200
  - batch_size: 16
  - notes: reasonable quality for fine-tuning when you have moderate data and a GPU

- `high` — long, high-quality fine-tuning
  - epochs: 400
  - batch_size: 8
  - notes: higher quality but much slower and needs more compute/VRAM

These map to parameters used when the script builds the training command. The `--epochs` override simply replaces the value taken from the preset. Other trainable parameters (see below) can be adjusted by editing the generated command or by passing flags if you run the Piper training CLI directly.

Tunable parameters (exposed / relevant)

- `--data.batch_size` (internal mapping) — increase to use more GPU memory and speed up per-step throughput. If you run out of VRAM, reduce.
- `--max_epochs` / `--epochs` — how many epochs to train. `quality` presets set defaults.
- `--model.sample_rate` — default 22050 Hz. Only change if your data uses a different SR and you know the model supports it.
- `--data.espeak_voice` — language/voice used to phonemize text via `espeak-ng` (e.g., `en-us`, `pl`); adjust if phonemization fails for your language.
- `--ckpt_path` — path to the base checkpoint to warm-start training. Using a checkpoint is strongly recommended.
- `--model.vocoder_warmstart_ckpt` — copy vocoder weights from a checkpoint without copying phoneme embeddings (useful when phoneme sets differ).
- `--data.num_symbols` / phoneme settings — advanced: only change if working with custom phoneme ids.
- `--device` — `cuda` or `cpu`. The script adds `--device cuda` when you pass `--gpu` to `train.py`.

Where to change advanced settings

When `train.py` prints the Piper CLI command, you can copy it and modify any parameter before running. The Piper training CLI (`python3 -m piper.train fit`) accepts a wide range of options — use `python3 -m piper.train fit --help` for full details.

Smaller checkpoints and targeted downloads

The Hugging Face dataset `rhasspy/piper-checkpoints` contains many checkpoints and may cause large downloads when snapshotting the whole dataset. To avoid large downloads:

- Inspect the dataset on the Hugging Face website and choose a specific checkpoint file you want (look for `medium` quality or language-specific filenames). If you find a smaller single file, note its repository path or model id.
- Use `train.py fetch-base --model-id <model-id-or-path>` to tell the script which specific file/repo to fetch. If a direct single-file download is available, the script will copy that smaller file into your `--dest-dir`.
- Alternatively, use the `huggingface_hub` tools manually to download a single file instead of a snapshot. For example (in Python):

```py
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="rhasspy/piper-checkpoints", filename="some_checkpoint.ckpt", repo_type="dataset", cache_dir="~/.piper/checkpoints")
```

- If you cannot find small single-file checkpoints, you can still use the snapshot approach and then delete unused files in the snapshot cache to reclaim space.

Targeted single-file downloads (recommended)

The `fetch-base` command now supports targeted single-file downloads to avoid snapshotting the whole dataset. Use either `--model-id` to supply a filename substring or `--model-pattern` to match parts of filenames (for example `en`, `pl`, `male`, or `medium`). The command will list candidate `.ckpt` files and let you choose one interactively, or use `--yes` to accept the first match non-interactively.

Example:

```bash
python3 train.py fetch-base --dest-dir ~/.piper/checkpoints/medium --quality medium --model-pattern en --yes
```

Notes:
- Set `HF_TOKEN` in your environment to increase rate limits and speed up downloads from Hugging Face.
- If no matching file is found the command will fall back to a full `snapshot_download` (the previous behavior).

Automation and convenience provided by `train.py`

- `init` — prepares normalized `wavs/` and `metadata.csv` from your `samples/<lang>/` directory.
- `fetch-base` — will attempt to install `huggingface_hub` inside the current venv (if missing) and download a checkpoint into the specified destination.
- `train` — will automatically call `fetch-base` if you do not provide `--ckpt`, placing the downloaded checkpoint under `~/.piper/checkpoints/<quality>/base_checkpoint.ckpt` and then printing the full `piper.train fit` command. Use `--run` to execute it.
- `export` — runs Piper's ONNX export helper after training.
- `synth-test` — uses the `piper` CLI to synthesize audio from an exported `.onnx` model.

Recommended end-to-end workflows

Scenario A — Quick experiment (CPU or small GPU):

```bash
# prepare
python3 train.py init --samples-dir samples_en --out-dir quick_exp --lang en --quality low

# fetch small checkpoint (if you have a specific small model id):
python3 train.py fetch-base --dest-dir ~/.piper/checkpoints/low --quality low --model-id "<your-small-model-id>"

# see training command
python3 train.py train --out-dir quick_exp --ckpt ~/.piper/checkpoints/low/base_checkpoint.ckpt --voice-name tomek --quality low

# if happy, run (be prepared for CPU-only training to be slow):
python3 train.py train --out-dir quick_exp --ckpt ~/.piper/checkpoints/low/base_checkpoint.ckpt --voice-name tomek --quality low --run
```

Scenario B — Medium-quality fine-tune (recommended for most users with a modest GPU):

```bash
python3 train.py init --samples-dir samples_en --out-dir training_en --lang en --quality medium
python3 train.py fetch-base --dest-dir ~/.piper/checkpoints/medium --quality medium
# inspect printed checkpoint path, then run training (or dry-run first)
python3 train.py train --out-dir training_en --ckpt ~/.piper/checkpoints/medium/base_checkpoint.ckpt --voice-name tomek --quality medium --run --gpu
```

Scenario C — High-quality / long run (high VRAM GPU recommended):

```bash
python3 train.py init --samples-dir samples_en --out-dir training_en --lang en --quality high
python3 train.py fetch-base --dest-dir ~/.piper/checkpoints/high --quality high
python3 train.py train --out-dir training_en --ckpt ~/.piper/checkpoints/high/base_checkpoint.ckpt --voice-name tomek --quality high --run --gpu
```

Vocoder warmstart and cross-language transfers

If you have a vocoder checkpoint you'd like to warmstart from, supply the `--model.vocoder_warmstart_ckpt` option to the Piper training CLI (you will need to edit the printed command or run `python3 -m piper.train fit` manually). This can speed up convergence and improve quality.

Practical tips and troubleshooting

- Disk space: Hugging Face snapshots can be multi-GB. Use `--model-id` to target a single-file smaller checkpoint when possible.
- VRAM: reduce batch sizes if you hit OOM; `medium` -> batch_size 16; `high` -> 8. If GPU memory is limited, use `--gpu` only when a CUDA GPU is available; otherwise training will be on CPU and very slow.
- Phonemization: if `espeak-ng` phonemes are unsuitable, prepare a `metadata.csv` with phoneme strings (see Piper docs) or change `--data.phoneme_type` during training.
- Missing deps: to run training from source you may need to install training extras: when inside the Piper repo run `python3 -m pip install -e '.[train]'` and follow `docs/TRAINING.md` for building C extensions.

Advanced: find smaller checkpoints on Hugging Face

1. Visit https://huggingface.co/datasets/rhasspy/piper-checkpoints and inspect contributors and filenames — some are smaller language-specific checkpoints.
2. Use `hf_hub_download` to download a single `.ckpt` file by filename instead of snapshotting the whole dataset.

Final notes

This workflow aims to make the minimal set of steps for preparing your voice dataset and fine-tuning a Piper voice. `train.py` automates dataset creation, base-checkpoint fetching, command generation, exporting and a synthesis test. For best results when cloning your own male voice in English and Polish, pick medium or high presets and use a base checkpoint (prefer a same-language checkpoint if available). If you want, I can now:

- add a minimal `voice_config.json` template that `train.py` writes into each `training_*` folder, or
- implement `train_and_export.sh` that runs training (dry-run first) and then exports and runs synth-test automatically.

Tell me which of those you prefer next.
