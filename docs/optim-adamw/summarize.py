#!/usr/bin/env python3
"""Print Markdown tables from measure.py JSON, without changing recorded observations."""

import argparse
import json
import math
from pathlib import Path
import re
import statistics


def read(path):
    data = json.loads(path.read_text())
    timings, memory, jit = {}, {}, {}
    for run in data["runs"]:
        if run["exit_code"]:
            raise ValueError(f"{path}: failed command {run['args']}")
        args = run["args"]
        if args[0] == "--verify-path":
            continue
        key = tuple(args[1:3])
        if args[0] == "--samples-json":
            rows = [json.loads(line) for line in run["stdout"].splitlines()
                    if line.startswith("{")]
            timings[key] = [row for row in rows if "sample" in row]
            if len(timings[key]) != data["samples"]:
                raise ValueError(f"{path}: incomplete samples for {key}")
        elif args[0] == "--measure-memory":
            values = dict(re.findall(r"(allocs|in_use|reserved|peak)=(\d+)", run["stdout"]))
            memory[key] = {k: int(v) for k, v in values.items()}
        elif args[0] == "--measure-jit":
            jit[key] = float(re.search(r"first_step_ms=([\d.]+)", run["stdout"])[1])
    return data, timings, memory, jit


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.files:
        data, timings, memory, jit = read(path)
        print(f"\n### {data['backend']}\n\n原始数据：[ {path.name} ](data/{path.name})；features `{data['features']}`。\n")
        print("| Workload | Mode | enqueue median ms | total median ms | total p95 ms | retained drop+sync median ms |")
        print("|---|---|---:|---:|---:|---:|")
        for (mode, workload), rows in timings.items():
            med = lambda k: statistics.median(row[k] for row in rows)
            print(f"| {workload} | {mode} | {med('enqueue_ms'):.4f} | {med('total_ms'):.4f} | "
                  f"{percentile([r['total_ms'] for r in rows], .95):.4f} | {med('retained_drop_sync_ms'):.4f} |")
        print("\n| Workload | Mode | live allocations | live MiB | pool peak upper bound MiB | first step ms |")
        print("|---|---|---:|---:|---:|---:|")
        for key, first in jit.items():
            mode, workload = key
            mem = memory.get(key, {})
            mib = lambda k: f"{mem[k] / 2**20:.3f}" if k in mem else "N/A"
            print(f"| {workload} | {mode} | {mem.get('allocs', 'N/A')} | {mib('in_use')} | {mib('peak')} | {first:.3f} |")


if __name__ == "__main__":
    main()
