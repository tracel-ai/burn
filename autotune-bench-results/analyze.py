#!/usr/bin/env python3
"""Scores the adaptive autotune scheduler against the fixed-sample one.

Reads the JSONL autotune records each configuration wrote and answers two questions per
autotune key:

  * cost   -- benchmarking wall clock, as the sum of the per-candidate TuningStep durations
              (compile + warmup + samples). It is split into the time spent on candidates that
              produced a measurement and the time spent on candidates that failed to launch,
              because only the former is work the scheduler can shorten.
  * choice -- which candidate was picked, priced against the `reference` run, which measures
              every candidate at the full fixed sample count with no short circuit.

Usage: analyze.py [round-dir ...]   (default: every round*/ subdirectory)
"""

import glob
import json
import os
import statistics
import sys
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = ["reference", "fixed", "adaptive", "floor"]
# Configurations that are real tuning strategies; `floor` is a cost floor, not a strategy.
STRATEGIES = ["reference", "fixed", "adaptive"]


def dur(d):
    """serde-serialized `Duration` -> seconds."""
    if d is None:
        return None
    return d["secs"] + d["nanos"] / 1e9


def key_id(key):
    definition = key.get("definition", {})
    analysis = key.get("analysis", {})
    return json.dumps([definition, analysis], sort_keys=True)


def key_label(key):
    definition = key.get("definition", key)
    analysis = key.get("analysis", {})
    parts = [f"{n}={definition[n]}" for n in ("m", "n", "k") if n in definition]
    parts += [str(analysis[n]) for n in ("kind", "scale_global") if n in analysis]
    return " ".join(parts) or key_id(key)[:70]


def load(round_dir, config):
    path = os.path.join(round_dir, config, "records.jsonl")
    if not os.path.exists(path):
        return None

    records = OrderedDict()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "key" not in record:
                print(f"warning: {path}: {record}", file=sys.stderr)
                continue

            ctx = record.get("log_context") or {}
            events = ctx.get("events") or []
            steps = [e["TuningStep"] for e in events if "TuningStep" in e]
            short = [e["ShortCircuit"] for e in events if "ShortCircuit" in e]

            medians, winner = {}, None
            for result in record["results"]:
                outcome = result["outcome"]
                if "Ok" not in outcome:
                    continue
                ok = outcome["Ok"]
                medians[ok["name"]] = dur(ok["computation"]["median"])
                if ok["index"] == record["fastest_index"]:
                    winner = ok["name"]

            # A step for a candidate that never produced a measurement is a launch/compile
            # failure: unavoidable cost the sampling strategy has no say over.
            measured_time = sum(d for name, d in ((n, dur(t)) for n, t in steps) if name in medians)
            failed_time = sum(d for name, d in ((n, dur(t)) for n, t in steps) if name not in medians)

            records[key_id(record["key"])] = {
                "label": key_label(record["key"]),
                "winner": winner,
                "medians": medians,
                "measured": len(medians),
                "candidates": len(record["results"]),
                "steps": len(steps),
                "measured_time": measured_time,
                "failed_time": failed_time,
                "bench_time": measured_time + failed_time,
                "short_circuit": short[0] if short else None,
            }
    return records


def pct(new, old):
    return 100 * (1 - new / old) if old else float("nan")


def main():
    rounds = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "remote-k620", "round*")))
    rounds = [r if os.path.isabs(r) else os.path.join(HERE, r) for r in rounds]
    rounds = [r for r in rounds if os.path.isdir(r)]
    if not rounds:
        sys.exit("no round directories found; run ./run.sh <round> first")

    loaded = []
    for round_dir in rounds:
        data = {config: load(round_dir, config) for config in CONFIGS}
        if any(v is None for v in data.values()):
            print(f"skipping incomplete {os.path.basename(round_dir)}", file=sys.stderr)
            continue
        loaded.append((os.path.basename(round_dir), data))

    print(f"rounds: {', '.join(name for name, _ in loaded)}")
    print()

    # ------------------------------------------------------------------ cost
    print("=" * 104)
    print("BENCHMARKING COST PER ROUND (seconds; 'live' = candidates that produced a measurement)")
    print("=" * 104)
    print(f"{'round':<10}{'config':<12}{'total':>10}{'live':>10}{'failed':>10}"
          f"{'vs fixed (total)':>20}{'vs fixed (live)':>20}")

    totals = defaultdict(list)
    lives = defaultdict(list)
    for name, data in loaded:
        fixed_total = sum(e["bench_time"] for e in data["fixed"].values())
        fixed_live = sum(e["measured_time"] for e in data["fixed"].values())
        for config in CONFIGS:
            total = sum(e["bench_time"] for e in data[config].values())
            live = sum(e["measured_time"] for e in data[config].values())
            failed = sum(e["failed_time"] for e in data[config].values())
            totals[config].append(total)
            lives[config].append(live)
            mark_total = "-" if config == "fixed" else f"{pct(total, fixed_total):+.1f}%"
            mark_live = "-" if config == "fixed" else f"{pct(live, fixed_live):+.1f}%"
            print(f"{name:<10}{config:<12}{total:>10.2f}{live:>10.2f}{failed:>10.2f}"
                  f"{mark_total:>20}{mark_live:>20}")

    print("-" * 104)
    print(f"{'MEAN':<10}{'config':<12}{'total':>10}{'live':>10}{'':>10}"
          f"{'vs fixed (total)':>20}{'vs fixed (live)':>20}")
    mean_total = {c: statistics.mean(totals[c]) for c in CONFIGS}
    mean_live = {c: statistics.mean(lives[c]) for c in CONFIGS}
    for config in CONFIGS:
        mark_total = "-" if config == "fixed" else f"{pct(mean_total[config], mean_total['fixed']):+.1f}%"
        mark_live = "-" if config == "fixed" else f"{pct(mean_live[config], mean_live['fixed']):+.1f}%"
        print(f"{'':<10}{config:<12}{mean_total[config]:>10.2f}{mean_live[config]:>10.2f}{'':>10}"
              f"{mark_total:>20}{mark_live:>20}")
    print()

    # ------------------------------------------------------ per-key cost
    print("=" * 104)
    print("BENCHMARKING COST PER KEY (mean over rounds, seconds)")
    print("=" * 104)
    print(f"{'autotune key':<46}{'reference':>11}{'fixed':>11}{'adaptive':>11}{'floor':>11}"
          f"{'saved':>10}{'reducible':>11}{'live':>6}")

    all_keys = OrderedDict()
    for _, data in loaded:
        for kid, entry in data["reference"].items():
            all_keys.setdefault(kid, entry["label"])

    for kid, label in all_keys.items():
        cells, live_counts = {}, []
        for config in CONFIGS:
            values = [data[config][kid]["bench_time"] for _, data in loaded if kid in data[config]]
            cells[config] = statistics.mean(values) if values else None
        live_counts = [data["reference"][kid]["measured"] for _, data in loaded if kid in data["reference"]]
        row = f"{label[:45]:<46}"
        for config in CONFIGS:
            row += f"{cells[config]:>11.3f}" if cells[config] is not None else f"{'-':>11}"
        if cells["fixed"] and cells["adaptive"]:
            row += f"{pct(cells['adaptive'], cells['fixed']):>+9.1f}%"
        else:
            row += f"{'-':>10}"
        # How much of the gap between `fixed` and the cost floor the adaptive scheduler closes.
        # This is the share of genuinely reducible time it recovers, with compilation excluded.
        reducible = None
        if cells["fixed"] and cells["floor"] is not None:
            gap = cells["fixed"] - cells["floor"]
            if gap > 0 and cells["adaptive"] is not None:
                reducible = (cells["fixed"] - cells["adaptive"]) / gap
        row += f"{100 * reducible:>10.0f}%" if reducible is not None else f"{'-':>11}"
        row += f"{max(live_counts) if live_counts else 0:>6}"
        print(row)
    print()

    # ---------------------------------------------------------------- picks
    print("=" * 104)
    print("PICK QUALITY (chosen candidate priced with the reference run's medians)")
    print("=" * 104)
    print(f"{'round':<9}{'autotune key':<40}{'config':<10}{'pick':<27}{'vs best':>16}")

    stats = {c: {"agree": 0, "total": 0, "losses": []} for c in ("fixed", "adaptive")}
    for name, data in loaded:
        reference = data["reference"]
        for kid, ref in reference.items():
            best, best_time = ref["winner"], ref["medians"].get(ref["winner"])
            for config in ("fixed", "adaptive"):
                entry = data[config].get(kid)
                if entry is None:
                    continue
                stats[config]["total"] += 1
                if entry["winner"] == best:
                    stats[config]["agree"] += 1
                    continue
                pick_time = ref["medians"].get(entry["winner"])
                if pick_time and best_time:
                    ratio = pick_time / best_time
                    stats[config]["losses"].append(ratio)
                    note = f"{100 * (ratio - 1):>+15.1f}%"
                else:
                    note = f"{'unpriced':>16}"
                print(f"{name:<9}{ref['label'][:39]:<40}{config:<10}{str(entry['winner'])[:26]:<27}{note}")

    print("-" * 104)
    for config in ("fixed", "adaptive"):
        s = stats[config]
        losses = s["losses"]
        detail = ""
        if losses:
            detail = (f"; when it differs: mean +{100 * (statistics.mean(losses) - 1):.1f}%, "
                      f"worst +{100 * (max(losses) - 1):.1f}%")
        print(f"{config:<10} matches the reference pick on {s['agree']}/{s['total']} key-rounds{detail}")

    # Disagreements the two configurations share are the short circuit's doing, not the
    # scheduler's: `fixed` runs the same short circuit with the old sampling strategy.
    print()
    shared = 0
    adaptive_only = 0
    for name, data in loaded:
        for kid, ref in data["reference"].items():
            f_pick = data["fixed"].get(kid, {}).get("winner")
            a_pick = data["adaptive"].get(kid, {}).get("winner")
            if a_pick != ref["winner"]:
                if f_pick == a_pick:
                    shared += 1
                else:
                    adaptive_only += 1
    print(f"adaptive mispicks shared with fixed (i.e. caused by the short circuit): {shared}")
    print(f"adaptive mispicks unique to the adaptive scheduler:                     {adaptive_only}")
    print()

    # ------------------------------------------------------------ stability
    print("=" * 104)
    print("PICK STABILITY ACROSS ROUNDS (how often each config lands on the same candidate)")
    print("=" * 104)
    print(f"{'autotune key':<46}" + "".join(f"{c:>19}" for c in STRATEGIES))
    for kid, label in all_keys.items():
        row = f"{label[:45]:<46}"
        for config in STRATEGIES:
            picks = [data[config][kid]["winner"] for _, data in loaded if kid in data[config]]
            distinct = len(set(picks))
            row += f"{('stable' if distinct == 1 else f'{distinct} different'):>19}"
        print(row)


if __name__ == "__main__":
    main()
