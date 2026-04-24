# Training guide — Piper voice cloning (quick examples + detailed options)

This file documents the simplified training workflow implemented by `train.py` in this
workspace. It includes quick usage examples, tunable parameters, and practical
instructions for small debug runs and full fine-tuning.

Files you will use

- `train.py` — the runner script (commands: `init`, `train`, `export`).
- `train.sh` — lightweight wrapper to run `train.py` from the repository root.
- `synth.py` — runtime synthesis script for ONNX models and checkpoints.
- `samples/` — your input folder with `en/` and/or `pl/` subfolders containing audio and optional transcripts.
- `training_*` — dataset folders created by `train.py init` containing `wavs/`, `metadata.csv`, and `README_TRAINING.md`.

## Quick usage examples

1) Prepare a dataset from your samples (normalize audio to 22050 Hz):

```bash
python3 train.py init --samples-dir samples_en --out-dir training_en --lang en
python3 train.py init --samples-dir samples_pl --out-dir training_pl --lang pl
```

If you don't have existing audio/text pairs you can build them interactively using
`record_samples.py`, which uses Whisper to transcribe your voice. See the project
README for usage.

2) Run a small debug training session using a local checkpoint from another drive:

```bash
python3 train.py train --out-dir training_en --ckpt /mnt/d/Audio/my-debug.ckpt --voice-name tomek --quality low --epochs 1 --run
```

This is the fastest way to verify the workflow with a tiny number of epochs and
an existing checkpoint. If `--ckpt` is omitted, `train.py` will automatically
fetch a base checkpoint into `~/.piper/checkpoints/<quality>`.

3) See the actual training command without executing it:

```bash
python3 train.py train --out-dir training_en --ckpt ~/.piper/checkpoints/medium/base_checkpoint.ckpt --voice-name tomek --quality medium
```

4) Run training for real:

```bash
python3 train.py train --out-dir training_en --ckpt ~/.piper/checkpoints/medium/base_checkpoint.ckpt --voice-name tomek --quality medium --run
```

You can also use the helper wrapper:

```bash
./train.sh train --out-dir training_en --voice-name tomek --quality medium --epochs 5 --run
```

5) Export the trained checkpoint to ONNX:

```bash
python3 train.py export --checkpoint training_en/tts_output/checkpoint_latest.ckpt --output training_en/tomek-medium.onnx
```

## Targeted checkpoint selection

When training without a local checkpoint, `train.py` can auto-fetch a base
checkpoint. If you want to target a specific file from the Hugging Face dataset,
use:

- `--model-id` — a filename or path substring to select a single checkpoint
- `--model-pattern` — a substring or regex to narrow the search
- `--yes` — accept the first match non-interactively

Example:

```bash
python3 train.py train --out-dir training_en --quality medium --model-pattern en --yes --run
```

## Debug and small-scale training

For quick tests, use the repository samples directory and a tiny epoch count:

```bash
python3 train.py train --out-dir training_en --ckpt /mnt/d/Audio/my-debug.ckpt --voice-name test --quality low --epochs 1 --run
```

This is useful when you want to verify that the dataset, checkpoint, and
training command all work together before committing to a longer run.

## Training options

The `train.py train` command exposes training-related tuning parameters:

- `--quality` — low/medium/high presets for epochs and batch size
- `--epochs` — override the preset total number of epochs
- `--batch-size` — override the preset batch size
- `--num-workers` — DataLoader workers
- `--shuffle-mode` — `strong`, `normal`, `weak`, or `off`
- `--sample-rate` — model sample rate (default 22050)
- `--learning-rate` — model learning rate
- `--learning-rate-d` — discriminator learning rate
- `--lr-decay` — generator learning rate decay
- `--lr-decay-d` — discriminator learning rate decay
- `--segment-size` — model segmentation window
- `--precision` — trainer precision, `16` or `32`
- `--espeak-voice` — override espeak phonemization voice
- `--ckpt` — checkpoint path to warm-start training
- `--model-id`, `--model-pattern`, `--yes` — select a remote checkpoint when auto-downloading

## Direct synthesis

After export, use `synth.py` for runtime synthesis:

```bash
python3 synth.py --model training_en/tomek-medium.onnx --text "Hello world" --out-file demo.wav
```

You can also synthesize directly from a checkpoint:

```bash
python3 synth.py --ckpt /mnt/d/Audio/my-debug.ckpt --text "Hello world" --out-file demo.wav
```

## Practical tips

- For very small test runs, use `--quality low` and `--epochs 1`.
- If you have a ready checkpoint on another disk, use `--ckpt /mnt/d/Audio/<file>.ckpt`.
- If the export fails, you can still use the checkpoint directly via `synth.py --ckpt`.
- Keep `tts_output` inside your dataset folder so training outputs are easy to find.

## Notes

`train.py` may automatically fetch a base checkpoint when `--ckpt` is omitted. It
also sanitizes downloaded checkpoints to remove extraneous Lightning metadata
that can interfere with training.
