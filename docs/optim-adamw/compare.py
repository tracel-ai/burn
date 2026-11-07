#!/usr/bin/env python3
"""Alternate saved AdamW binaries to separate stage effects from time-varying load."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--workloads", nargs="+", default=[
        "tiny", "mixed_rank", "large_1", "large_4", "large_16", "long_short",
    ])
    args = parser.parse_args()
    binaries = {"before": args.before.resolve(), "after": args.after.resolve()}
    result = {"backend": args.backend, "features": args.features, "shared": args.shared,
              "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": [], "binaries": {}}
    for label, path in binaries.items():
        manifest = json.loads(Path(str(path) + ".json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["binary_sha256"]:
            raise ValueError(f"{path}: manifest mismatch")
        if set(manifest["features"].split(",")) != set(args.features.split(",")):
            raise ValueError(f"{path}: features differ from the saved build manifest")
        result["binaries"][label] = manifest
    env = {**os.environ, "BURN_DEVICE": args.backend}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for workload in args.workloads:
        for round in range(4):
            order = ["before", "after"] if round % 2 == 0 else ["after", "before"]
            for label in order:
                command = [str(binaries[label]), "--samples-json", "fused", workload]
                if args.shared:
                    command.append("--shared")
                load_before = os.getloadavg()
                run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=300)
                result["runs"].append({"stage": label, "round": round, "workload": workload,
                                       "load_before": load_before, "load_after": os.getloadavg(),
                                       "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                if run.returncode:
                    raise RuntimeError(run.stderr[-4000:])
            print(args.backend, workload, round, "ok", flush=True)


if __name__ == "__main__":
    main()
