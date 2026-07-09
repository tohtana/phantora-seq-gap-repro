# Shared-Loop 4-GPU Phantora Repro

This is the handoff package for the full-model sequence-length mismatch. It
contains one Python training loop and one launcher. The real and Phantora rows
run the same `full_model_zero3_repro.py`; only `--backend real` versus
`--backend phantora` and the required launcher differ.

Both paths use Qwen3-8B dimensions, bf16, DeepSpeed ZeRO-3, full activation
checkpointing, AdamW (`torch_adam`, `lr=5e-5`), synthetic causal-LM batches,
the same warmup/measurement policy, and SDPA. Both set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid allocator
fragmentation changing the 4096-token feasibility result. SDPA is deliberate: the current
Phantora runtime cannot run FlashAttention-2's variable-length path with the
simulated integer inputs needed by micro-batch 4.

## Requirements

- one node with four H100 GPUs;
- Phantora installed at `PHANTORA_HOME` (default `/opt/phantora`), including
  `tests/phantora_utils.py` and `tests/docker/deepspeed/config_gen.py`;
- a Python environment with PyTorch, DeepSpeed, and Transformers. Set `VENV`
  to it.

`phantora_utils` is not copied here. It is imported from the Phantora runtime
only in `--backend phantora` mode for its simulated timer, function tracer, and
DeepSpeed patch.

## Run

```bash
VENV=/path/to/venv PHANTORA_HOME=/opt/phantora \
  ./run_4gpu.sh
```

Set `INSTALL_HOST_PACKAGES=1` only on base images that still need
`libopenblas-dev` before importing Phantora's paired PyTorch wheel.

Defaults are `MODEL=8B`, `SEQ_LENS=512,1024,2048,4096`,
`MICRO_BATCH_SIZE=4`, `GRAD_ACCUM=16`, `WARMUP=2`, and `ITERATIONS=3`.
The launcher writes `real.jsonl`, `sim.jsonl`, raw logs, and `summary.md` to
`$OUT_DIR` (or a timestamped directory below the current directory).
Each JSONL row includes the SHA-256 of `full_model_zero3_repro.py`; the real
and simulated rows must carry the same value.

For an allocator-control run that leaves the variable unset, pass an explicit
empty value:

```bash
PYTORCH_CUDA_ALLOC_CONF='' VENV=/path/to/venv PHANTORA_HOME=/opt/phantora \
  SEQ_LENS=512,1024,2048 ./run_4gpu.sh
```

## Validated Results

Both runs used Qwen3-8B dimensions, bf16, DeepSpeed ZeRO-3, full activation
checkpointing, AdamW, SDPA, 4 H100 GPUs, micro-batch 4, gradient accumulation
16, warmup 2, and 3 measured iterations.

### `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

| seq | real tok/s | Phantora sim tok/s | sim/real | real step s | sim step s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 12,147 | 8,268 | 0.68 | 10.79 | 15.85 |
| 1024 | 21,136 | 16,485 | 0.78 | 12.40 | 15.90 |
| 2048 | 25,211 | 28,138 | 1.12 | 20.80 | 18.63 |
| 4096 | 20,537 | 32,014 | 1.56 | 51.06 | 32.75 |

### Allocator override unset

The control Job passes `PYTORCH_CUDA_ALLOC_CONF=''`; the launcher removes the
variable from the real, Phantora server, and Phantora runner processes.

| seq | real tok/s | Phantora sim tok/s | sim/real | real step s | sim step s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 12,158 | 8,045 | 0.66 | 10.78 | 16.29 |
| 1024 | 20,815 | 16,269 | 0.78 | 12.59 | 16.11 |
| 2048 | 25,255 | 28,251 | 1.12 | 20.76 | 18.56 |

The no-override real run OOMs at seq 4096, so that row has no comparable
Phantora value. The two tables show the same short-sequence underestimation and
2048-token overestimation; the override makes 4096 feasible but is not the
cause of the direction change.

The resulting table is valid evidence of a common training loop only when the
summary records `attention_impl=sdpa` in both JSONL files. The earlier
FlashAttention-2-real/SDPA-sim result is intentionally not used as this
package's same-operator comparison.
