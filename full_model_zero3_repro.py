#!/usr/bin/env python3
"""Shared real/Phantora ZeRO-3 full-model training step.

The real and Phantora runs invoke this exact file with only ``--backend``
changed.  The model, ZeRO-3 config, synthetic batches, activation
checkpointing, warmup, and measured forward/backward/step loop are shared.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM


MODEL_DIMS = {
    "8B": (36, 4096, 12288, 32, 8, 151936),
    "14B": (40, 5120, 17408, 40, 8, 151936),
}


class RandomTokens(Dataset):
    """Shape-identical synthetic causal-LM batches for both backends."""

    def __init__(self, vocab_size, sequence_length, count):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, _index):
        tokens = torch.randint(0, self.vocab_size, (self.sequence_length,))
        return tokens, tokens.clone()


class Timer:
    def __init__(self, backend):
        self.backend = backend
        self._enable_trace = None
        self._disable_trace = None
        if backend == "phantora":
            # This is Phantora's bundled test helper, not part of this package.
            from phantora_utils import (
                disable_function_tracer,
                enable_function_tracer,
                install_phantora_deepspeed_patches,
                time_pair,
            )

            install_phantora_deepspeed_patches()
            self._now = lambda: time_pair()[0]
            self._enable_trace = enable_function_tracer
            self._disable_trace = disable_function_tracer
        else:
            self._now = time.perf_counter

    def start(self):
        if self._enable_trace is not None:
            self._enable_trace()

    def stop(self):
        if self._disable_trace is not None:
            self._disable_trace()

    def now(self):
        return self._now()


def build_deepspeed_config(args, world_size):
    return {
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "train_batch_size": args.micro_batch_size * args.gradient_accumulation * world_size,
        "steps_per_print": 1000000,
        "optimizer": {
            "type": "AdamW",
            "params": {"torch_adam": True, "lr": 5e-5},
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 0.0,
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": False,
            "contiguous_gradients": False,
        },
        "wall_clock_breakdown": False,
    }


def build_model(args, ds_config):
    layers, hidden, ffn_hidden, heads, kv_heads, vocab_size = MODEL_DIMS[args.model]
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=ffn_hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=args.sequence_length,
        use_cache=False,
        attn_implementation=args.attention_impl,
        torch_dtype=torch.bfloat16,
    )

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        # This construction path is deliberately shared by real and simulated
        # ZeRO-3 so parameter partitioning is not a backend-specific difference.
        with deepspeed.zero.Init(config_dict_or_path=ds_config, dtype=torch.bfloat16):
            model = LlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(original_dtype)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model, vocab_size


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    timer = Timer(args.backend)
    ds_config = build_deepspeed_config(args, world_size)
    model, vocab_size = build_model(args, ds_config)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[parameter for parameter in model.parameters() if parameter.requires_grad],
        config=ds_config,
    )

    batch_count = (args.warmup + args.iterations + 1) * args.gradient_accumulation
    dataset = RandomTokens(vocab_size, args.sequence_length, batch_count * args.micro_batch_size)
    data_iter = iter(DataLoader(dataset, batch_size=args.micro_batch_size))

    def train_step():
        for _ in range(args.gradient_accumulation):
            input_ids, labels = next(data_iter)
            output = engine(
                input_ids=input_ids.to(engine.device),
                labels=labels.to(engine.device),
            )
            engine.backward(output.loss)
            engine.step()

    timer.start()
    try:
        for _ in range(args.warmup):
            train_step()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()

        step_times = []
        for _ in range(args.iterations):
            started = timer.now()
            train_step()
            torch.cuda.synchronize()
            step_times.append(timer.now() - started)
    finally:
        timer.stop()

    if rank == 0:
        step_time = sum(step_times) / len(step_times)
        tokens_per_step = (
            args.micro_batch_size
            * args.gradient_accumulation
            * world_size
            * args.sequence_length
        )
        record = {
            "record_type": "phantora_full_model_zero3_repro",
            "training_script": Path(__file__).name,
            "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "backend": args.backend,
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "model": args.model,
            "attention_impl": args.attention_impl,
            "world_size": world_size,
            "zero_stage": 3,
            "gradient_checkpointing": "full",
            "sequence_length": args.sequence_length,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "warmup_steps": args.warmup,
            "measured_steps": args.iterations,
            "step_time_s": step_time,
            "tokens_per_step": tokens_per_step,
            "tokens_per_second": tokens_per_step / step_time,
            "per_step_s": step_times,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 * 1024),
        }
        print("REPRO_RESULT " + json.dumps(record, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("real", "phantora"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_DIMS), default="8B")
    parser.add_argument("--attention-impl", choices=("sdpa",), default="sdpa")
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0)
    main(parser.parse_args())
