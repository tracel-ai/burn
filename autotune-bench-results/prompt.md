I need you to run an autotune benchmark on this machine's GPU and report the results.

## Setup

Clone or Workspace these three repos as siblings in the same parent directory (the path
dependencies between them rely on this layout):

    <parent>/
      burn/
      cubecl/
      cubek/

    git clone https://github.com/tracel-ai/burn
    git clone https://github.com/tracel-ai/cubecl
    git clone https://github.com/tracel-ai/cubek

Check out these branches:

  - burn    -> test/adaptive-autotune-bench
  - cubecl  -> feat/adaptive-autotune-scheduler
  - cubek   -> main

## Patch dependencies to local paths

Both burn and cubek default to git dependencies. Switch each to the local path
block so the branches above are what actually gets compiled.

In `burn/Cargo.toml`, under `[workspace.dependencies]`: comment out the four
entries under `### For the main burn branch. ###` (cubecl, cubecl-common,
cubecl-zspace, cubek) and uncomment the four under
`### For local development. ###`.

In `cubek/Cargo.toml`, do the same: comment out the two entries under
`### For the main cubek branch. ###` and uncomment the two under
`### For local development. ###`.

Verify with `cargo tree -p cubecl | head` that cubecl resolves to a local path
and not a git rev. Do not skip this — if it still resolves to git, the whole
run measures the wrong code.

## Build and run

    cd burn
    cargo build --release -p autotune-adaptive-bench

    cd autotune-bench-results
    ./run.sh round1 reference fixed adaptive adaptive_nosc floor
    ./run.sh round2 adaptive reference fixed floor adaptive_nosc
    ./run.sh round3 fixed adaptive reference adaptive_nosc floor

The config order is rotated between rounds on purpose, so no configuration is
always the one paying for a cold GPU. Each config runs in its own process with
the autotune cache off and writes `autotune.log`, `records.jsonl`, and
`results.csv` into `round<N>/<config>/`.

Then:

    python3 analyze.py

  - GPU model, driver version, and CUDA version (`nvidia-smi`).
  - The full `analyze.py` output.
  - The `round*/` directories, zipped.