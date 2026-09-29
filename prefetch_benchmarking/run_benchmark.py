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

"""Sequential, exclusive-output benchmark launcher for this experiment."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent


def main() -> None:
    """Launch requested runs, stopping on foreign GPU use or a failed run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "timing", "torch", "nsys"], required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="Unique name:forward-depth pairs")
    parser.add_argument("--ep", type=int, default=4)
    args = parser.parse_args()
    source = ROOT / "examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["BENCH_MODE"] = args.mode
    env["BENCH_WARMUP"] = "2" if args.mode == "smoke" else "30"
    env["BENCH_STEPS"] = "8" if args.mode == "smoke" else "100" if args.mode == "timing" else "3"
    for spec in args.runs:
        name, depth_text = spec.rsplit(":", 1)
        depth = int(depth_text)
        if Path(name).name != name or depth < 0:
            raise ValueError("Expected a directory basename and nonnegative depth")
        pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).splitlines()
        foreign = {pid.strip() for pid in pids if pid.strip()} - {"405242"}
        if foreign:
            raise RuntimeError(f"Other GPU processes present; not launching: {sorted(foreign)}")
        run_dir = ROOT / name
        run_dir.mkdir(exist_ok=False)
        env["BENCH_RUN_DIR"] = str(run_dir)
        config = yaml.safe_load(source.read_text())
        config["step_scheduler"].update(
            max_steps=int(env["BENCH_WARMUP"]) + int(env["BENCH_STEPS"]),
            val_every_steps=1000000,
            validate_on_checkpoint=False,
        )
        config["checkpoint"].update(enabled=False, checkpoint_dir=str(run_dir / "checkpoints"))
        config["distributed"].update(
            enable_fsdp2_prefetch=True,
            fsdp2_forward_prefetch_depth=depth,
            ep_size=args.ep,
        )
        config["mfu"] = {"peak_tflops": 989.0}
        config_path = run_dir / "config.yaml"
        with config_path.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        command = [
            "uv",
            "run",
            "--no-project",
            "--active",
            "--no-sync",
            "automodel",
            str(config_path),
            "--nproc-per-node",
            "8",
        ]
        if args.mode == "nsys":
            command = [
                env.get("BENCH_NSYS", "nsys"),
                "profile",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--cpuctxsw=none",
                "--force-overwrite=false",
                f"--output={run_dir / 'nsys'}",
            ] + command
        manifest = {
            "command": command,
            "mode": args.mode,
            "depth": depth,
            "ep": args.ep,
            "allowed_external_pid": 405242,
            "start_utc": time.time(),
        }
        with (run_dir / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)
        print(f"START {name} depth={depth} ep={args.ep} mode={args.mode}", flush=True)
        start = time.perf_counter()
        with (run_dir / "gpu-telemetry.csv").open("x") as telemetry, (run_dir / "console.log").open("x") as log:
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu",
                    "--format=csv",
                    "-l",
                    "1",
                ],
                stdout=telemetry,
                stderr=subprocess.STDOUT,
            )
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        completion = {
            "returncode": result.returncode,
            "process_elapsed_s": time.perf_counter() - start,
            "end_utc": time.time(),
        }
        with (run_dir / "completion.json").open("x") as stream:
            json.dump(completion, stream, indent=2)
        print(f"END {name} {completion}", flush=True)
        if result.returncode:
            raise RuntimeError(f"Run failed; inspect {run_dir / 'console.log'}")
        windows = list(run_dir.glob("window-rank*.json"))
        if len(windows) != 8:
            raise RuntimeError(f"Expected 8 completed rank measurements, found {len(windows)}")


if __name__ == "__main__":
    main()
