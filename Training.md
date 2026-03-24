# Training tuning & GPU performance handbook

This document explains the performance-oriented parameters used by the `Training.ipynb` notebook and the `piper` training launcher. It is written for someone new to machine learning who wants to maximize GPU utilization while preserving model quality. Read through the short primer, then use the concrete tuning steps and examples for incremental testing on your T4 GPU.

**Quick links**
- Notebook: `Training.ipynb`

**Goals**
- Understand what the main parameters do.
- Know how each parameter affects GPU memory, throughput, and training quality.
- Follow a safe, incremental tuning procedure to increase GPU utilization without causing OOM or degraded training.

**Audience**: novice ML practitioners running `piper` training in Colab (T4) or similar GPUs.

---

**1. Core parameters — what they are and why they matter**

- `BATCH_SIZE`
  - What: Number of training examples processed in one forward/backward pass.
  - Performance effect: Directly increases GPU memory use and per-step computation. Larger batch sizes usually increase throughput (samples/sec) because the GPU work is amortized across more samples.
  - Quality effect: Changes effective optimization dynamics. Very large batch sizes can require learning-rate scaling or more epochs for similar generalization.
  - Practical: If you increase `BATCH_SIZE`, watch for Out Of Memory (OOM) errors. Use gradient accumulation (below) if memory is limiting.

- `SEGMENT_SIZE` (notebook passes to `--model.segment_size`)
  - What: Number of audio samples (or model timesteps) per training segment. Larger segments produce longer contexts per sample.
  - Performance effect: Increasing `SEGMENT_SIZE` increases memory per sample and per-batch compute. It can improve throughput if your GPU has unused compute capacity, but it also rises memory usage quickly.
  - Quality effect: Larger segments can help the model learn longer temporal dependencies; too small segments might reduce audio coherence.

- `PRECISION` (`'16'` or `'32'`) — mixed precision
  - What: Numeric precision used for model weights/activations during training.
  - Performance effect: `16` (AMP / mixed precision) reduces memory and increases throughput on GPUs with Tensor Cores (e.g., T4). `32` uses more memory but may be numerically more stable (rarely needed for SOTA speech models).
  - Quality effect: Generally none for properly configured AMP; however, monitor loss spikes and use dynamic loss-scaling if available.

- `NUM_WORKERS` (DataLoader)
  - What: Number of background worker processes used for data loading and preprocessing.
  - Performance effect: Affects CPU->GPU pipeline throughput. Too few workers can make the GPU idle waiting for batches; too many wastes CPU and memory.
  - Practical: On Colab T4, 2–4 is a good starting point. On multi-core machines increase accordingly.

- `ACCUMULATE_GRAD_BATCHES` (gradient accumulation)
  - What: Accumulates gradients for N steps before applying an optimizer step. Effective batch size = `BATCH_SIZE * ACCUMULATE_GRAD_BATCHES`.
  - Performance effect: Allows larger effective batch sizes without additional GPU memory. Throughput per optimizer step decreases (more forward/backward passes), but you can reproduce large-batch training dynamics safely.
  - Quality effect: With correct learning-rate scaling, results are similar to using a larger real batch.

- `--trainer.devices`, `--trainer.accelerator`
  - What: Tells Lightning to use GPU(s) and how many.
  - Performance effect: Using the correct accelerator and device count ensures GPU use. Multi-GPU requires distributed setup.

- `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF`
  - What: Environment variables controlling allocator behavior (split sizes, fragmentation behavior).
  - Performance effect: Tweaking these can reduce fragmentation and OOM frequency on long-running runs.

- `torch.backends.cudnn.benchmark`
  - What: When `True`, cuDNN times multiple convolution algorithms and picks the fastest for given sizes.
  - Performance effect: Can improve throughput on fixed-size inputs but may increase startup time and non-determinism.

---

**2. Which parameters affect GPU memory vs compute vs IO**

- GPU memory (largest effect): `BATCH_SIZE`, `SEGMENT_SIZE`, `PRECISION` (32 vs 16), model size (number of parameters), optimizer states.
- GPU compute (throughput): `BATCH_SIZE`, `SEGMENT_SIZE`, `PRECISION`, `NUM_WORKERS` (indirectly), data pipeline efficiency.
- IO/CPU bottlenecks: `NUM_WORKERS`, dataset caching, `CACHE_DIR`, feature precomputation (mel spectrograms), disk speed.

Tip: If GPU memory is low but utilization is low (e.g., 5GB used, 5–20% utilization), the bottleneck is likely IO/CPU — increase `NUM_WORKERS`, precompute caches, or increase `BATCH_SIZE` gradually.

---

**3. Safe incremental tuning procedure (recommended)**

1. Baseline: run the training with the notebook defaults (`BATCH_SIZE` from section 2, `NUM_WORKERS=0`, `SEGMENT_SIZE=4096`, `PRECISION='16'`) and confirm a successful step or two.
2. Monitor the GPU and memory in a separate cell/terminal:

```bash
watch -n 2 nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv
```

or in Python:

```py
import torch
print('allocated', torch.cuda.memory_allocated()/1e9, 'GB')
print('reserved', torch.cuda.memory_reserved()/1e9, 'GB')
```

3. If GPU memory is low but utilization also low (<30%):
   - Increase `NUM_WORKERS` from 0 to 2, rerun. If GPU utilization increases, keep it.
   - Ensure `CACHE_DIR` is set and precompute spectrograms if the dataset pipeline spends time on CPU.

4. If GPU memory is underused but utilization low and data pipeline is fast:
   - Increase `BATCH_SIZE` stepwise: multiply by 2 (e.g., 4 → 8 → 16) until either utilization improves or you hit OOM.
   - If OOM occurs, drop back one step and instead set `ACCUMULATE_GRAD_BATCHES` to 2 to double effective batch without more memory.

5. If the GPU memory is available but utilization still low, increase `SEGMENT_SIZE` in steps (e.g., 4096 → 6144 → 8192). Watch memory growth closely.

6. Try changing `PRECISION`:
   - If you need more headroom for `BATCH_SIZE`, ensure `PRECISION='16'` to reduce memory.
   - If you want to squeeze more throughput and the model is stable, mixed precision often improves speed on T4.

7. If you want to reproduce larger-batch training dynamics, set `ACCUMULATE_GRAD_BATCHES` and scale learning rate accordingly (linear scaling rule: lr * effective_batch_size / base_batch_size).

8. Final validation: ensure training loss and validation metrics behave similarly after tuning. Minor tuning of learning rate may be required when changing `BATCH_SIZE` significantly.

---

**4. Concrete recommended starting values (T4 / Colab)**

- Conservative, low risk
  - `BATCH_SIZE = 4` or `8`
  - `SEGMENT_SIZE = 4096`
  - `NUM_WORKERS = 2`
  - `PRECISION = '16'`
  - `ACCUMULATE_GRAD_BATCHES = 1` (use 2 if you want larger effective batch without memory change)

- Moderate (recommended to test)
  - `BATCH_SIZE = 8` or `16`
  - `SEGMENT_SIZE = 6144`
  - `NUM_WORKERS = 2-4`
  - `PRECISION = '16'`
  - `ACCUMULATE_GRAD_BATCHES = 1-2`

- Aggressive (higher OOM risk)
  - `BATCH_SIZE = 16+`
  - `SEGMENT_SIZE = 8192+`
  - `NUM_WORKERS = 4+`
  - `PRECISION = '32'` (only if you have the memory headroom)

Notes: T4 has 16GB of RAM — the sweet spot often lies in BATCH_SIZE 8–16 and SEGMENT_SIZE 4096–8192 with AMP enabled.

---

**5. Advanced knobs and behavior**

- `accumulate_grad_batches` and learning-rate scaling
  - If you double effective batch size, scale learning rate roughly by 2 (linear scaling). Prefer conservative increases and watch validation.

- CUDNN benchmarking
  - `torch.backends.cudnn.benchmark = True` can improve speed when input sizes are constant. However, it may increase initial CPU time and cause non-deterministic results. Use if you have fixed segment sizes.

- `PYTORCH_ALLOC_CONF` / fragmentation
  - Example: `export PYTORCH_ALLOC_CONF=max_split_size_mb:128` reduces fragmentation by forcing splits at a lower threshold.
  - Use when you see repeated small allocations causing fragmentation and OOM over time.

- `CACHE_DIR` and precomputation
  - Precompute mel spectrograms and cache them to `CACHE_DIR` to avoid repeated CPU/IO processing. This often yields big GPU utilization improvements.

- Disk / IO considerations
  - Keep the dataset on fast storage (Colab local disk `/content` is faster than Drive). If you store audio on Drive, consider copying needed files to local temp storage before training.

---

**6. Quick diagnostics and small helper snippets**

- Check GPU utilization and memory (shell):

```bash
watch -n 2 nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv
```

- Check PyTorch memory details (notebook cell):

```py
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print('allocated (GB)', torch.cuda.memory_allocated()/1e9)
    print('reserved  (GB)', torch.cuda.memory_reserved()/1e9)
```

- Basic probe loop (very cautious) to find max safe `BATCH_SIZE` without OOM:

```py
import torch
def find_safe_batch(model_init, dataloader_factory, start=2, max_try=32):
    b = start
    while b <= max_try:
        try:
            dl = dataloader_factory(batch_size=b)
            model = model_init()
            # run single forward/backward on one batch to test
            x, y = next(iter(dl))
            out = model(x.cuda())
            loss = (out - y.cuda()).abs().mean()
            loss.backward()
            print('safe batch', b)
            return b
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                b = b // 2 if b>2 else b+1
                b *= 2
            else:
                raise
    return None
```

Note: the probe must be used cautiously and adapted to your dataset and model; avoid running it on a long training job.

---

**7. Practical examples: modify `Training.ipynb` section 2**

Set the tuning variables near the top of the notebook so they propagate:

```py
BATCH_SIZE = 8
NUM_WORKERS = 2
SEGMENT_SIZE = 6144
PRECISION = '16'
ACCUMULATE_GRAD_BATCHES = 1
```

In section 3 (the launcher), ensure the CLI args pick up these variables, for example:

```py
cli_args += ['--data.batch_size', str(BATCH_SIZE),
             '--data.num_workers', str(NUM_WORKERS),
             '--model.segment_size', str(SEGMENT_SIZE),
             '--trainer.precision', PRECISION,
             '--trainer.accumulate_grad_batches', str(ACCUMULATE_GRAD_BATCHES)]
```

---

**8. Checklist when tuning**

- Start from defaults and change only one variable per experiment.
- Monitor both GPU memory and GPU utilization (`nvidia-smi`).
- If OOM occurs, try: reduce `BATCH_SIZE`, reduce `SEGMENT_SIZE`, switch to `PRECISION='16'`, or use `ACCUMULATE_GRAD_BATCHES`.
- If GPU is underutilized: increase `NUM_WORKERS`, precompute cache, increase `BATCH_SIZE` or `SEGMENT_SIZE` carefully.
- Re-evaluate model metrics after tuning—throughput gains are worthless if validation degrades.

---

**9. Final notes and common pitfalls**

- Don't conflate GPU memory usage with utilization: having mem free doesn't guarantee compute is saturated; check `utilization.gpu`.
- Mixed precision (`'16'`) is usually safe and yields the best speed/memory trade-off on T4.
- Data pipeline bottlenecks are very common—use `NUM_WORKERS` and caching aggressively.
- Use `accumulate_grad_batches` instead of blindly increasing `BATCH_SIZE` when memory-limited.

---

**Glossary (acronyms & short terms)**

- OOM — Out Of Memory: the GPU runs out of available memory for allocations; causes runtime errors during forward/backward or optimizer steps.
- AMP — Automatic Mixed Precision: using 16-bit fused/conditional math to reduce memory and improve throughput while keeping numerical stability (often via PyTorch AMP).
- GPU — Graphics Processing Unit: accelerator used to run model computations faster than CPU.
- CPU — Central Processing Unit: general-purpose processor; used for data loading, preprocessing, and some parts of training.
- IO — Input/Output: disk/network operations (reading audio files, writing caches) that can bottleneck training if slow.
- cuDNN — CUDA Deep Neural Network library: NVIDIA library that implements efficient primitives for convolutions/RNNs; `torch.backends.cudnn.benchmark` interacts with it.
- T4 — NVIDIA T4 GPU: a common Colab GPU with 16GB memory and Tensor Cores suited for mixed precision.
- ONNX — Open Neural Network Exchange: portable format for exporting models to run outside PyTorch.
- LR — Learning Rate: a hyperparameter controlling the size of optimizer steps; often scaled when changing batch size.

Acknowledgements: distilled from common PyTorch/PyTorch Lightning best-practices and tuned for the `piper` training workflow in `Training.ipynb`.
