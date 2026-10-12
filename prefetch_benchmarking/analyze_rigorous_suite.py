# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Analyze paired benchmark artifacts and write a persistent JSON report."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRATCH_ROOT = Path("/dev/shm/automodel-prefetch-bench")


def _read_runs() -> list[dict]:
    runs = []
    for manifest_path in sorted(SCRATCH_ROOT.glob("*/manifest.json")):
        run_dir = manifest_path.parent
        completion_path = run_dir / "completion.json"
        summary_path = run_dir / "summary.json"
        if not completion_path.exists() or not summary_path.exists():
            continue
        completion = json.loads(completion_path.read_text())
        if completion["returncode"] != 0:
            continue
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads(summary_path.read_text())
        summary.update(manifest)
        summary.update(completion)
        summary["tokens_per_second"] = (
            summary["global_batch_size"] * summary["seq_len"] / summary["avg_iter_time_seconds"]
        )
        runs.append(summary)
    return runs


def _smoke_results(runs: list[dict]) -> dict:
    grouped = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        if run["phase"] == "smoke":
            run_index = run.get("run_index")
            if run_index is None:
                match = re.search(r"[AB](\d+)$", run["name"])
                if match is None:
                    raise ValueError(f"Cannot infer smoke index from {run['name']!r}")
                run_index = int(match.group(1))
            grouped[(run["model"], run["ep_size"], run["seq_len"])][run_index][run["condition"]] = run
    results = {}
    for key, repeats in grouped.items():
        pairs = []
        for repeat_index, conditions in sorted(repeats.items()):
            if conditions.keys() != {"A", "B"}:
                continue
            baseline = conditions["A"]["losses"]
            treatment = conditions["B"]["losses"]
            pairs.append(
                {
                    "repeat": repeat_index,
                    "losses_match_bitwise_as_json_floats": baseline == treatment,
                    "max_loss_difference": max(abs(a - b) for a, b in zip(baseline, treatment, strict=True)),
                    "baseline_losses": baseline,
                    "prefetch_losses": treatment,
                }
            )
        if not pairs:
            continue
        baseline_repeats = [repeats[index]["A"]["losses"] for index in sorted(repeats) if "A" in repeats[index]]
        prefetch_repeats = [repeats[index]["B"]["losses"] for index in sorted(repeats) if "B" in repeats[index]]

        def repeat_difference(losses: list[list[float]]) -> float | None:
            if len(losses) < 2:
                return None
            reference = losses[0]
            return max(abs(a - b) for other in losses[1:] for a, b in zip(reference, other, strict=True))

        results[f"{key[0]}-ep{key[1]}-s{key[2]}"] = {
            "repeats": len(pairs),
            "all_pairs_bitwise_equal": all(pair["losses_match_bitwise_as_json_floats"] for pair in pairs),
            "max_cross_condition_loss_difference": max(pair["max_loss_difference"] for pair in pairs),
            "baseline_repeat_max_loss_difference": repeat_difference(baseline_repeats),
            "prefetch_repeat_max_loss_difference": repeat_difference(prefetch_repeats),
            "all_losses_finite": all(
                math.isfinite(value)
                for pair in pairs
                for losses_key in ("baseline_losses", "prefetch_losses")
                for value in pair[losses_key]
            ),
            "pairs": pairs,
        }
    return results


def _timing_results(runs: list[dict], bootstrap_samples: int) -> dict:
    grouped = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        if run["phase"] == "timing":
            key = (run["model"], run["ep_size"], run["seq_len"])
            run_index = run.get("run_index")
            if run_index is None:
                match = re.search(r"[AB](\d+)$", run["name"])
                if match is None:
                    raise ValueError(f"Cannot infer pair index from {run['name']!r}")
                run_index = int(match.group(1))
            grouped[key][run_index][run["condition"]] = run

    output = {}
    generator = np.random.default_rng(2026)
    for key, pairs_by_index in grouped.items():
        pairs = []
        for pair_index in sorted(pairs_by_index):
            conditions = pairs_by_index[pair_index]
            if conditions.keys() == {"A", "B"}:
                pairs.append((conditions["A"], conditions["B"]))
        if not pairs:
            continue
        ratios = np.array(
            [treatment["avg_iter_time_seconds"] / baseline["avg_iter_time_seconds"] for baseline, treatment in pairs]
        )
        boot = generator.choice(np.log(ratios), size=(bootstrap_samples, len(pairs)), replace=True).mean(axis=1)
        baseline_latencies = [baseline["avg_iter_time_seconds"] for baseline, _ in pairs]
        treatment_latencies = [treatment["avg_iter_time_seconds"] for _, treatment in pairs]
        baseline_mfu = [baseline["avg_mfu_percent"] for baseline, _ in pairs]
        treatment_mfu = [treatment["avg_mfu_percent"] for _, treatment in pairs]
        output[f"{key[0]}-ep{key[1]}-s{key[2]}"] = {
            "pairs": len(pairs),
            "baseline_median_step_ms": 1000 * statistics.median(baseline_latencies),
            "prefetch_median_step_ms": 1000 * statistics.median(treatment_latencies),
            "median_latency_reduction_percent": 100
            * (1 - statistics.median(treatment_latencies) / statistics.median(baseline_latencies)),
            "paired_geometric_latency_reduction_percent": 100 * (1 - float(np.exp(np.log(ratios).mean()))),
            "paired_latency_reduction_95pct_bootstrap_ci": [
                float(value) for value in np.quantile(100 * (1 - np.exp(boot)), [0.025, 0.975])
            ],
            "baseline_median_mfu_percent": statistics.median(baseline_mfu),
            "prefetch_median_mfu_percent": statistics.median(treatment_mfu),
            "median_mfu_relative_gain_percent": 100
            * (statistics.median(treatment_mfu) / statistics.median(baseline_mfu) - 1),
            "baseline_median_tokens_per_second": statistics.median(
                baseline["tokens_per_second"] for baseline, _ in pairs
            ),
            "prefetch_median_tokens_per_second": statistics.median(
                treatment["tokens_per_second"] for _, treatment in pairs
            ),
            "peak_allocated_gib": max(run["peak_allocated_gib"] for pair in pairs for run in pair),
            "peak_reserved_gib": max(run["peak_reserved_gib"] for pair in pairs for run in pair),
            "pair_latency_ratios": ratios.tolist(),
            "runs": [run for pair in pairs for run in pair],
        }
    return output


def main() -> None:
    """Write the combined smoke, timing, and artifact inventory report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    runs = _read_runs()
    report = {
        "scratch_root": str(SCRATCH_ROOT),
        "run_count": len(runs),
        "smoke": _smoke_results(runs),
        "timing": _timing_results(runs, args.bootstrap_samples),
        "profiles": [
            {
                "name": run["name"],
                "model": run["model"],
                "condition": run["condition"],
                "depth": run["depth"],
            }
            for run in runs
            if run["phase"] == "profile"
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
