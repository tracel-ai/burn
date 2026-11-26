"""Replay cubecl's patience over measured times in priority order: how many are measured, and the
regret of the round's winner against the best of all."""
import csv, sys
from collections import defaultdict

IMPROVEMENT = 0.98  # cubecl's TimeChange: an improvement is more than 2% faster than the leader


def replay(times, min_measured, max_misses):
    leader, misses = None, 0
    for n, t in enumerate(times, 1):
        if leader is None or t < leader * IMPROVEMENT:
            leader, misses = t, 0
        else:
            misses += 1
        if n >= min_measured and misses >= max_misses:
            return n, min(times[:n])
    return len(times), min(times)


rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['us'] not in ('', 'nan')]
key = sys.argv[2] if len(sys.argv) > 2 else 'seq'
by = defaultdict(list)
for i, r in enumerate(rows):
    by[r[key]].append((i, r))
for k, rs in by.items():
    order = sorted(rs, key=lambda t: (-int(t[1]['priority']), t[0]))
    listed = sorted(rs, key=lambda t: t[0])
    best = min(float(r['us']) for _, r in rs)
    for name, seq in (('priority', order), ('list', listed)):
        times = [float(r['us']) for _, r in seq]
        cells = []
        for mm, misses in ((10, 5), (10, 10), (20, 10)):
            n, got = replay(times, mm, misses)
            cells.append(f"P({mm},{misses}): {n:>3} measured, {got / best:.3f}x")
        print(f"{key}={k:>5} {name:<8} " + " | ".join(cells))
