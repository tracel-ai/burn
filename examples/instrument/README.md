# Instrument

The example shows you how to:

- Mark a model's components with `#[compute]`, and price the ones that declare what they must move
  and compute with `Bounds`.
- Probe the device's peaks with `Peaks`, time a clean pass, and collect one pass with
  `Instrument`.
- Save the pass for `burn-instrument-report`, which scores each component against the wall it is
  bound by.

The model is a stack of 4 residual feed-forward blocks over 512 tokens of width 1024, with a hidden
width of 4096. Each projection and the whole stack declare their bounds; the norm, the activation
and the residual add are left unpriced for the fuser to fold into the projections.

## Running

Collect a pass, then print its report:

```bash
cargo run --release -p instrument -- pass.json
cargo run --release -p burn-instrument-report -- pass.json
```

The pass is written to `pass.json` when no path is given. The device is `BURN_DEVICE`'s, or the
default one of the linked runtime.

The collector reads a CubeCL runtime's launches, so one has to be linked. `wgpu` is the default; to
use another, replace it:

```bash
cargo run --release -p instrument --no-default-features --features rocm -- pass.json
```

The runtimes are `wgpu`, `vulkan`, `metal`, `cuda`, `rocm` and `cpu`.

## Comparing passes

Save a pass, change something, collect again, and hold the new pass against the saved one:

```bash
cargo run --release -p instrument -- was.json
# change the model, the backend, or the code under it
cargo run --release -p instrument -- now.json
cargo run --release -p burn-instrument-report -- now.json --against was.json
```

The comparison refuses two passes that ran on different devices, one fused and the other unfused,
one with its kernels timed in place and the other alone, or with a clean run measured on only one
side, and says which.
