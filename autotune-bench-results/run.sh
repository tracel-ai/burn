#!/usr/bin/env bash
# Drives examples/autotune-adaptive-bench once per configuration.
#
# One process = one configuration, so each one tunes every key from cold and writes its own
# autotune.log / records.jsonl / results.csv. cubecl finds its settings by walking up from the
# working directory, so each configuration runs with its own directory as cwd and picks up the
# cubecl.toml written there.
#
# Usage: ./run.sh <round> [configs...]     (default: all five configs, in listed order)
#   e.g. ./run.sh round1
#        ./run.sh round2 adaptive reference fixed
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

BIN="${BENCH_BIN:-$REPO/target/release/autotune-adaptive-bench}"
DTYPE="${BENCH_DTYPE:-f32}"

ROUND="${1:-round1}"
shift || true
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(reference fixed adaptive adaptive_nosc floor)
fi

if [ ! -x "$BIN" ]; then
  echo "missing $BIN" >&2
  echo "build it first: cargo build --release -p autotune-adaptive-bench" >&2
  exit 1
fi

# Per configuration: adaptive, short_circuit, min_samples, max_samples.
config_params() {
  case "$1" in
    # Ground truth: every candidate at the full fixed sample count, nothing dropped early.
    reference)     echo "off off 10 10" ;;
    # The production baseline today: fixed sample count plus the short circuit.
    fixed)         echo "off on 3 10" ;;
    # The new scheduler.
    adaptive)      echo "on on 3 10" ;;
    # Isolates the round robin from the short circuit.
    adaptive_nosc) echo "on off 3 10" ;;
    # Cost floor: one sample per candidate. Not a strategy, just a lower bound.
    floor)         echo "on on 1 1" ;;
    *) echo "unknown config: $1" >&2; exit 1 ;;
  esac
}

echo "round=$ROUND configs=${CONFIGS[*]} dtype=$DTYPE"

for config in "${CONFIGS[@]}"; do
  read -r adaptive short_circuit min_samples max_samples <<<"$(config_params "$config")"

  out="$HERE/$ROUND/$config"
  rm -rf "$out"
  mkdir -p "$out"

  cat >"$out/cubecl.toml" <<EOF
[autotune.logger]
level = "full"
file = "$out/autotune.log"
append = false

[autotune.recorder]
file = "$out/records.jsonl"
append = false

[autotune.bench]
min_samples = $min_samples
max_samples = $max_samples
EOF

  echo "--- $config (adaptive=$adaptive short_circuit=$short_circuit samples=$min_samples..$max_samples)"

  # cwd = $out so cubecl picks up the cubecl.toml written just above.
  (
    cd "$out"
    BENCH_LABEL="$config" \
    BENCH_DTYPE="$DTYPE" \
    CUBECL_AUTOTUNE_CACHE=off \
    CUBECL_AUTOTUNE_SHORT_CIRCUIT="$short_circuit" \
    CUBECL_AUTOTUNE_BENCH_ADAPTIVE="$adaptive" \
      "$BIN" >"$out/results.csv" 2>"$out/stderr.log"
  )

  tail -1 "$out/stderr.log"
done

echo
echo "done. analyze with: python3 $HERE/analyze.py"
