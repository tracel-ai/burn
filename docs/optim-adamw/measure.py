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
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--module", action="store_true", help="Measure complete ModuleOptimizer steps")
    parser.add_argument("--repeats", type=int, default=1, help="Independent processes per workload/mode")
    parser.add_argument("--workloads", nargs="+", default=[
        "tiny", "mixed_rank", "large_1", "large_4", "large_16", "long_short",
    ])
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--source-manifest", type=Path,
                        help="Build manifest (default: <binary>.json, produced by build.py)")
    args = parser.parse_args()
    args.modes = args.modes or (["paired"] if args.module else ["reference", "fused"])
    allowed = {"paired", "fused", "foreach", "profile-fused", "profile-foreach"} if args.module else {"reference", "fused"}
    if args.repeats < 1 or not set(args.modes) <= allowed or (args.module and args.shared):
        parser.error("Invalid modes, repeat count, or shared module measurement")
    if args.module and "paired" in args.modes and {"fused", "foreach"} & set(args.modes):
        parser.error("Measure paired and single-model throughput in separate reports")
    binary = str(args.binary.resolve())
    manifest = json.loads((args.source_manifest or Path(binary + ".json")).read_text())
    with Path(binary).open("rb") as executable:
        binary_hash = hashlib.file_digest(executable, "sha256").hexdigest()
    if manifest["binary_sha256"] != binary_hash:
        raise ValueError("Source manifest belongs to another binary")
    if set(manifest["features"].split(",")) != set(args.features.split(",")):
        raise ValueError("Features differ from the saved build manifest")
    env = {**os.environ, "BURN_DEVICE": args.backend}
    report = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(), "backend": args.backend,
        "load_average": os.getloadavg(),
        "features": args.features, "binary_sha256": binary_hash,
        "module": args.module, "repeats": args.repeats, "shared": args.shared,
        "samples": 30, "paired_samples_per_mode": 40,
        "build": manifest, "sources": manifest["sources"], "runs": [],
    }
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

    if not args.module and "fused" in args.modes:
        run("--verify-path")
    for workload in args.workloads:
        for repeat in range(args.repeats):
            for mode in args.modes if repeat % 2 == 0 else reversed(args.modes):
                if args.module:
                    run("--module-samples", mode, workload)
                else:
                    run("--samples-json", mode, workload, *(["--shared"] if args.shared else []))
        if not args.module and not args.shared:
            for mode in args.modes:
                run("--measure-memory", mode, workload)
                run("--measure-jit", mode, workload)


if __name__ == "__main__":
    main()
