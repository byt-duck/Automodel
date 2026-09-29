# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment-only measurement support; imported only by the copied source tree."""

import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def save_trainable(models: list[torch.nn.Module], label: str) -> None:
    """Save rank-local trainable parameters and available gradients for comparison.

    Args:
        models: Sharded model modules. Parameter and gradient tensors retain their
            per-rank local shapes and axis order; DTensors are converted to local
            tensors without gathering. Snapshots own detached CPU copies.
        label: Unique snapshot label within this run.
    """
    if os.environ.get("BENCH_MODE") != "smoke":
        return
    tensors = {}
    for index, model in enumerate(models):
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            for suffix, tensor in (("parameter", parameter), ("gradient", parameter.grad)):
                if tensor is None:
                    continue
                local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
                tensors[f"{index}.{name}.{suffix}"] = local.detach().cpu().clone()
    path = Path(os.environ["BENCH_RUN_DIR"]) / f"{label}-rank{dist.get_rank()}.pt"
    with path.open("xb") as stream:
        torch.save(tensors, stream)


def record_prefetch(model: torch.nn.Module, edges: list) -> None:
    """Record the actual configured module edges once, before training."""
    names = {id(module): name for name, module in model.named_modules()}
    payload = [
        {"source": names[id(source)], "targets": [names[id(target)] for target in targets]} for source, targets in edges
    ]
    path = Path(os.environ["BENCH_RUN_DIR"]) / f"prefetch-rank{dist.get_rank()}.json"
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)


class Window:
    """Time complete optimizer iterations with synchronization only at window boundaries."""

    def __init__(self) -> None:
        self.rank = dist.get_rank()
        self.output = Path(os.environ["BENCH_RUN_DIR"])
        self.mode = os.environ.get("BENCH_MODE", "timing")
        self.warmup = int(os.environ.get("BENCH_WARMUP", "30"))
        self.steps = int(os.environ.get("BENCH_STEPS", "100"))
        self.count = 0
        self.records = []
        self.host_intervals = []
        self.fingerprints = []
        self.start = None

    def after_step(self, sample, batches: list[dict]) -> None:
        """Observe a completed step, including its logging and garbage collection.

        Args:
            sample: Recipe MetricsSample; scalar metrics may be device tensors.
            batches: Microbatches with input_ids shaped [batch, sequence] and
                labels shaped [batch, sequence]. Only shapes are read in timing
                runs; smoke runs copy inputs to CPU for reproducibility hashes.
        """
        self.count += 1
        if self.mode == "smoke":
            digest = hashlib.sha256()
            for batch in batches:
                for key in ("input_ids", "labels"):
                    digest.update(batch[key].detach().cpu().contiguous().numpy().tobytes())
            self.fingerprints.append(digest.hexdigest())
        if self.start is not None:
            now = time.perf_counter()
            self.host_intervals.append(now - self.previous)
            self.previous = now
            self.records.append((sample.metrics, sum(batch["input_ids"].numel() for batch in batches)))
        if self.count == self.warmup:
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start_utc = time.time()
            self.start = self.previous = time.perf_counter()
        if self.count == self.warmup + self.steps:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - self.start
            end_utc = time.time()
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            elapsed_tensor = torch.tensor(elapsed, device="cuda", dtype=torch.float64)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            metrics = {
                "rank": self.rank,
                "mode": self.mode,
                "warmup_steps": self.warmup,
                "measured_steps": len(self.records),
                "elapsed_local_s": elapsed,
                "elapsed_max_s": elapsed_tensor.item(),
                "start_utc": self.start_utc,
                "end_utc": end_utc,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
                "host_step_intervals_s": self.host_intervals,
                "global_useful_tokens": sum(float(record[0]["num_tokens_per_step"]) for record in self.records),
                "global_label_tokens": sum(float(record[0]["num_label_tokens"]) for record in self.records),
                "global_formula_flops": sum(float(record[0]["bench_flops"]) for record in self.records),
                "local_padded_tokens": sum(record[1] for record in self.records),
                "input_fingerprints": self.fingerprints,
            }
            with (self.output / f"window-rank{self.rank}.json").open("x") as stream:
                json.dump(metrics, stream, indent=2)
