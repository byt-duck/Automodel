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

"""Run paired FSDP2+EP forward-prefetch benchmarks on eight GPUs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path("/dev/shm/automodel-prefetch-bench")
TRACE_ROOT = REPOSITORY / "traces" / "rigorous-prefetch"
PYTHON = REPOSITORY / ".venv/bin/python"

MODEL_CONFIGS = {
    "nemotron": REPOSITORY / "examples/llm_benchmark/nemotron/nemotron_nano_v3_te_deepep.yaml",
    "qwen3": REPOSITORY / "examples/llm_benchmark/qwen/qwen3_moe_30b_te_deepep.yaml",
    "gptoss": REPOSITORY / "examples/llm_benchmark/gpt_oss/gptoss_20b_te_deepep.yaml",
}


def _gpu_processes() -> list[str]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def _write_config(
    source: Path,
    destination: Path,
    *,
    depth: int,
    ep_size: int,
    seq_len: int,
    warmup_steps: int,
    measured_steps: int,
    summary_path: Path,
) -> None:
    config = yaml.safe_load(source.read_text())
    config["benchmark"].update(
        warmup_steps=warmup_steps,
        peak_tflops=989,
        json_output_path=str(summary_path),
        nsys_start=-1,
        nsys_end=-1,
        nsys_ranks=[],
    )
    config["step_scheduler"].update(
        global_batch_size=8,
        local_batch_size=1,
        max_steps=warmup_steps + measured_steps,
        gc_every_steps=1000000,
    )
    config["dataset"].update(seq_len=seq_len, num_samples=100000)
    config["checkpoint"]["enabled"] = False
    config["distributed"].update(
        ep_size=ep_size,
        tp_size=1,
        cp_size=1,
        pp_size=1,
        enable_fsdp2_prefetch=True,
        fsdp2_forward_prefetch_depth=depth,
        fsdp2_backward_prefetch_depth=0,
    )
    config["distributed"].setdefault("moe", {})["reshard_after_forward"] = False
    # Keep the benchmark self-contained on installations without DeepEP/UCCL-EP.
    # This still exercises the same nested EP+FSDP topology and FSDP all-gathers;
    # only token dispatch uses the native DTensor collectives.
    config["model"].setdefault("backend", {}).update(
        attn="flex" if source.name.startswith("gptoss") else "sdpa",
        dispatcher="torch",
        experts="torch_mm",
    )
    destination.write_text(yaml.safe_dump(config, sort_keys=False))


def _run_one(
    *,
    model: str,
    phase: str,
    condition: str,
    depth: int,
    run_index: int,
    ep_size: int,
    seq_len: int,
    warmup_steps: int,
    measured_steps: int,
) -> None:
    run_name = f"{model}-ep{ep_size}-s{seq_len}-{phase}-{condition}{run_index}"
    run_dir = SCRATCH_ROOT / run_name
    completion_path = run_dir / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if completion.get("returncode") == 0:
            print(f"SKIP completed {run_name}", flush=True)
            return
        archived = run_dir.with_name(f"{run_dir.name}.failed-{int(time.time())}")
        run_dir.rename(archived)
        print(f"ARCHIVE failed run as {archived.name}", flush=True)

    processes = _gpu_processes()
    if processes:
        raise RuntimeError(f"GPUs are not exclusive; found: {processes}")

    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.yaml"
    _write_config(
        MODEL_CONFIGS[model],
        config_path,
        depth=depth,
        ep_size=ep_size,
        seq_len=seq_len,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        summary_path=summary_path,
    )

    cache_root = SCRATCH_ROOT / "cache"
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(REPOSITORY),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(cache_root / "huggingface"),
        HF_HUB_CACHE=str(cache_root / "huggingface/hub"),
        XDG_CACHE_HOME=str(cache_root / "xdg"),
        TMPDIR=str(cache_root / "tmp"),
        TRITON_CACHE_DIR=str(cache_root / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache_root / "torchinductor"),
    )
    nvidia_package_root = REPOSITORY / ".venv/lib/python3.12/site-packages/nvidia"
    nvidia_library_paths = sorted(path for path in nvidia_package_root.glob("*/lib") if path.is_dir())
    env["LD_LIBRARY_PATH"] = ":".join(
        [*(str(path) for path in nvidia_library_paths), env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(":")
    for path in (
        Path(env["HF_HUB_CACHE"]),
        Path(env["XDG_CACHE_HOME"]),
        Path(env["TMPDIR"]),
        Path(env["TRITON_CACHE_DIR"]),
        Path(env["TORCHINDUCTOR_CACHE_DIR"]),
    ):
        path.mkdir(parents=True, exist_ok=True)
    if phase == "profile":
        env["AUTOMODEL_TORCH_PROFILE_DIR"] = str(run_dir)
    else:
        env.pop("AUTOMODEL_TORCH_PROFILE_DIR", None)

    command = [str(PYTHON), "-m", "nemo_automodel.cli.app", str(config_path), "--nproc-per-node", "8"]
    manifest = {
        "name": run_name,
        "model": model,
        "phase": phase,
        "condition": condition,
        "depth": depth,
        "run_index": run_index,
        "ep_size": ep_size,
        "seq_len": seq_len,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "command": command,
        "source_config": str(MODEL_CONFIGS[model]),
        "start_time_unix": time.time(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"START {run_name}", flush=True)
    start = time.perf_counter()
    with (run_dir / "console.log").open("x") as log, (run_dir / "gpu-telemetry.csv").open("x") as telemetry:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu",
                "--format=csv",
                "-l",
                "1",
            ],
            stdout=telemetry,
            stderr=subprocess.STDOUT,
        )
        try:
            result = subprocess.run(command, cwd=REPOSITORY, env=env, stdout=log, stderr=subprocess.STDOUT)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)

    completion = {
        "returncode": result.returncode,
        "process_elapsed_seconds": time.perf_counter() - start,
        "end_time_unix": time.time(),
    }
    completion_path.write_text(json.dumps(completion, indent=2))
    print(f"END {run_name}: {completion}", flush=True)
    if result.returncode:
        raise RuntimeError(f"Run failed; inspect {run_dir / 'console.log'}")
    if not summary_path.exists():
        raise RuntimeError(f"Run did not create {summary_path}")

    if phase == "profile":
        persistent = TRACE_ROOT / run_name
        persistent.mkdir(parents=True, exist_ok=False)
        for filename in ("trace-rank0.json", "kernels-rank0.txt", "summary.json", "config.yaml", "manifest.json"):
            source = run_dir / filename
            if not source.exists():
                raise RuntimeError(f"Profile run did not create {source}")
            shutil.copy2(source, persistent / filename)


def main() -> None:
    """Run requested phases in a restartable, order-balanced sequence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "timing", "profile", "all"], default="all")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=sorted(MODEL_CONFIGS))
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--smoke-repeats", type=int, default=1)
    args = parser.parse_args()

    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    phases = [args.phase] if args.phase != "all" else ["smoke", "timing", "profile"]

    for model in args.models:
        if "smoke" in phases:
            for run_index in range(1, args.smoke_repeats + 1):
                for condition, depth in (("A", 0), ("B", 1)):
                    _run_one(
                        model=model,
                        phase="smoke",
                        condition=condition,
                        depth=depth,
                        run_index=run_index,
                        ep_size=args.ep,
                        seq_len=args.seq_len,
                        warmup_steps=2,
                        measured_steps=5,
                    )

        if "timing" in phases:
            orientations = [("A", 0, "B", 1), ("B", 1, "A", 0)]
            for pair_index in range(1, args.pairs + 1):
                orientation = orientations[(pair_index - 1) % len(orientations)]
                for condition, depth in ((orientation[0], orientation[1]), (orientation[2], orientation[3])):
                    _run_one(
                        model=model,
                        phase="timing",
                        condition=condition,
                        depth=depth,
                        run_index=pair_index,
                        ep_size=args.ep,
                        seq_len=args.seq_len,
                        warmup_steps=30,
                        measured_steps=100,
                    )

        if "profile" in phases:
            for condition, depth in (("A", 0), ("B", 1)):
                _run_one(
                    model=model,
                    phase="profile",
                    condition=condition,
                    depth=depth,
                    run_index=1,
                    ep_size=args.ep,
                    seq_len=args.seq_len,
                    warmup_steps=10,
                    measured_steps=3,
                )


if __name__ == "__main__":
    main()
