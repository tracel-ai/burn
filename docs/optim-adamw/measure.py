#!/usr/bin/env python3
"""Record reproducible AdamW samples; run each backend serially on an idle machine."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--modes", nargs="+", default=["reference", "fused"])
    parser.add_argument("--workloads", nargs="+", default=[
        "tiny", "mixed_rank", "large_1", "large_4", "large_16", "long_short",
    ])
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--source-manifest", type=Path,
                        help="Source hashes saved alongside this exact built binary")
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    env = {**os.environ, "BURN_DEVICE": args.backend}
    report = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(), "backend": args.backend,
        "load_average": os.getloadavg(),
        "features": args.features, "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "warmup_steps": 5, "samples": 30, "shared": args.shared,
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "sources": {}, "runs": [],
    }
    for path in [
        "Cargo.lock", "crates/burn-optim/benches/adamw.rs",
        "crates/burn-optim/Cargo.toml", "crates/burn-optim/build.rs",
        "crates/burn-optim/src/optim/adamw.rs",
        "crates/burn-optim/src/optim/adamw/cube.rs",
        "crates/burn-cubecl/src/kernel/optim/adamw.rs",
    ]:
        report["sources"][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if args.source_manifest:
        manifest = json.loads(args.source_manifest.read_text())
        if manifest["binary_sha256"] != report["binary_sha256"]:
            raise ValueError("Source manifest belongs to another binary")
        report["sources"] = manifest["sources"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def run(*options):
        command = [binary, *options]
        load_before = os.getloadavg()
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=300)
        record = {"args": list(options), "exit_code": result.returncode,
                  "load_before": load_before, "load_after": os.getloadavg(),
                  "stdout": result.stdout, "stderr": result.stderr}
        report["runs"].append(record)
        save()
        if result.returncode:
            raise RuntimeError(f"{command}: exit {result.returncode}\n{result.stderr[-4000:]}")
        print(args.backend, " ".join(options), "ok", flush=True)

    if "fused" in args.modes:
        run("--verify-path")
    for workload in args.workloads:
        for mode in args.modes:
            run("--samples-json", mode, workload, *(["--shared"] if args.shared else []))
            if not args.shared:
                run("--measure-memory", mode, workload)
                run("--measure-jit", mode, workload)


if __name__ == "__main__":
    main()
