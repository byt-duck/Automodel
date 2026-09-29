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

"""Analyze the saved experiment artifacts without using GPUs."""

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def _smoke() -> dict:
    report = {"input_hashes_match": True, "comparisons": {}}
    for rank in range(8):
        a_window = json.loads((ROOT / "smoke-A" / f"window-rank{rank}.json").read_text())
        b_window = json.loads((ROOT / "smoke-B" / f"window-rank{rank}.json").read_text())
        report["input_hashes_match"] &= a_window["input_fingerprints"] == b_window["input_fingerprints"]
    for label in ("first-gradient", "final"):
        for kind in ("parameter", "gradient"):
            stats = {
                "count": 0,
                "all_bitwise_equal": True,
                "finite": True,
                "max_abs": 0.0,
                "diff_sq": 0.0,
                "reference_sq": 0.0,
            }
            for rank in range(8):
                a = torch.load(ROOT / "smoke-A" / f"{label}-rank{rank}.pt", map_location="cpu", weights_only=True)
                b = torch.load(ROOT / "smoke-B" / f"{label}-rank{rank}.pt", map_location="cpu", weights_only=True)
                if a.keys() != b.keys():
                    raise ValueError("Trainable snapshot keys differ")
                for name in a:
                    if not name.endswith(f".{kind}"):
                        continue
                    x, y = a[name].double(), b[name].double()
                    if x.shape != y.shape:
                        raise ValueError(f"Shape mismatch: {name}")
                    stats["count"] += x.numel()
                    stats["all_bitwise_equal"] &= torch.equal(x, y)
                    stats["finite"] &= bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                    if x.numel():
                        stats["max_abs"] = max(stats["max_abs"], (x - y).abs().max().item())
                        stats["diff_sq"] += (x - y).square().sum().item()
                        stats["reference_sq"] += x.square().sum().item()
            stats["relative_l2"] = math.sqrt(stats["diff_sq"] / max(stats["reference_sq"], 1e-300))
            report["comparisons"][f"{label}.{kind}"] = stats
    a_logs = [json.loads(line) for line in (ROOT / "smoke-A/checkpoints/training.jsonl").read_text().splitlines()]
    b_logs = [json.loads(line) for line in (ROOT / "smoke-B/checkpoints/training.jsonl").read_text().splitlines()]
    if len(a_logs) != len(b_logs):
        raise ValueError("Smoke run lengths differ")
    report["max_loss_difference"] = max(abs(a["loss"] - b["loss"]) for a, b in zip(a_logs, b_logs))
    return report


def _run(path: Path) -> dict:
    ranks = [json.loads((path / f"window-rank{rank}.json").read_text()) for rank in range(8)]
    manifest = json.loads((path / "manifest.json").read_text())
    completion = json.loads((path / "completion.json").read_text())
    sample = ranks[0]
    duration = sample["elapsed_max_s"]
    if any(rank["global_useful_tokens"] != sample["global_useful_tokens"] for rank in ranks):
        raise ValueError("Ranks disagree on global token count")
    result = {
        "name": path.name,
        "mode": manifest["mode"],
        "depth": manifest["depth"],
        "ep": manifest["ep"],
        "elapsed_s": duration,
        "step_ms": 1000 * duration / sample["measured_steps"],
        "useful_tokens": sample["global_useful_tokens"],
        "useful_tps": sample["global_useful_tokens"] / duration,
        "label_tps": sample["global_label_tokens"] / duration,
        "padded_tps": sum(rank["local_padded_tokens"] for rank in ranks) / duration,
        "formula_flops": sample["global_formula_flops"],
        "full_training_equivalent_mfu_pct": 100 * sample["global_formula_flops"] / (8 * 989e12 * duration),
        "peak_allocated_gib": max(rank["peak_allocated_bytes"] for rank in ranks) / 2**30,
        "peak_reserved_gib": max(rank["peak_reserved_bytes"] for rank in ranks) / 2**30,
        "process_s": completion["process_elapsed_s"],
        "host_p50_ms": 1000 * float(np.quantile(sample["host_step_intervals_s"], 0.5)),
        "host_p95_ms": 1000 * float(np.quantile(sample["host_step_intervals_s"], 0.95)),
    }
    telemetry = []
    with (path / "gpu-telemetry.csv").open() as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            timestamp = datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
            if sample["start_utc"] <= timestamp <= sample["end_utc"]:
                telemetry.append((timestamp, int(row[1]), *[float(value.strip().split()[0]) for value in row[2:]]))
    if telemetry:
        result["mean_gpu_util_pct"] = statistics.mean(row[2] for row in telemetry)
        result["mean_gpu_power_w"] = statistics.mean(row[5] for row in telemetry)
        result["mean_sm_clock_mhz"] = statistics.mean(row[6] for row in telemetry)
        result["max_temperature_c"] = max(row[8] for row in telemetry)
        result["estimated_node_gpu_energy_j"] = 8 * result["mean_gpu_power_w"] * duration
    return result


def main() -> None:
    """Print numerical summaries or correctness evidence as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.smoke:
        report = _smoke()
    else:
        runs = [
            _run(path.parent)
            for path in sorted(ROOT.glob("*/completion.json"))
            if json.loads(path.read_text())["returncode"] == 0 and (path.parent / "window-rank7.json").exists()
        ]
        report = {"runs": runs}
        primary = {run["name"]: run for run in runs if run["name"].startswith("timing-")}
        if len(primary) == 12:
            pairs = [(primary[f"timing-A{i}"], primary[f"timing-B{i}"]) for i in range(1, 7)]
            if any(
                a["useful_tokens"] != b["useful_tokens"] or a["formula_flops"] != b["formula_flops"] for a, b in pairs
            ):
                raise ValueError("Primary pairs have unequal work")
            ratios = np.array([b["elapsed_s"] / a["elapsed_s"] for a, b in pairs])
            boot = np.random.default_rng(2026).choice(np.log(ratios), size=(20000, 6), replace=True).mean(axis=1)
            report["primary"] = {
                "baseline_mean_step_ms": statistics.mean(a["step_ms"] for a, _ in pairs),
                "prefetch_mean_step_ms": statistics.mean(b["step_ms"] for _, b in pairs),
                "paired_latency_reduction_pct": 100 * (1 - float(np.exp(np.log(ratios).mean()))),
                "paired_latency_reduction_95pct_bootstrap_ci": [
                    float(x) for x in np.quantile(100 * (1 - np.exp(boot)), [0.025, 0.975])
                ],
                "paired_throughput_gain_pct": 100 * (float(np.exp(-np.log(ratios).mean())) - 1),
                "pair_time_ratios": ratios.tolist(),
            }
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
