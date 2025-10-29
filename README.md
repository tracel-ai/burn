<div align="center">
<img src="./assets/logo-burn-neutral.webp" width="350px"/>

---

**Born in the backroom of a Jersey data center, Gaba Burn is the next generation tensor library.**

**It was handcrafted by a brain running solely on 5,000 pounds of premium gabagool.**

***The legacy of Gaba Burn lives in Rust.***

<img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcGwyZWd2M3Uzamh4aWs4amcybTh3dWluOHlwbmI5aHQ0cHhiZG5nciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l1J9Qha6bqbCs31q8/giphy.gif" width="400px"/>

---
</div>

<div align="left">

Gaba Burn is a pragmatic, performance-first fork of Burn focused on three things:

- Rock-solid, portable CPU performance (autovectorized Rust fallback + optional hand-tuned native kernels).
- Reproducible developer workflows: fast benches, deterministic fixtures, and xtask-driven experiments.
- Clear upgrade paths for GPU acceleration (CUDA/ROCm/Metal) while keeping local-first inference simple.

This repository is where we iterate quickly on native kernels, quantized primitives, and a small, auditable
set of runtime features so you can ship models that run well everywhere from laptops to clouds.

## What changed

- Native kernels: we now provide an optional, feature-gated path to build small, high-performance
  native kernels implemented in Zig. The `gaba-native-kernels` crate contains a prototype GEMM that
  can be built automatically when you enable the `zig` feature. The Rust crate always ships a
  correct triple-loop reference implementation so CI and casual contributors don't need Zig.
- Benchmarks: microbenchmarks are driven by `criterion` and live next to kernels. Use `cargo bench`
  (or our `xtask` harness) to get reproducible measurements across hosts.
- Vector search & embeddings: we've added a CPU-parallel vector search implementation (Rayon) with
  deterministic fixtures so retrieval experiments are repeatable and auditable.

## Native kernels (Zig)

Why Zig? Zig gives us a predictable, small toolchain for writing low-level, C-ABI kernels and calling
them from Rust. We use Zig to prototype carefully tuned inner loops (GEMM, small convolutions, q-matmul).

How it works:

1. The `gaba-native-kernels` crate contains `native/gemm.zig` and a `build.rs` that will invoke Zig
   when the crate is built with `--features zig`.
2. When `zig` is enabled, `build.rs` compiles `native/gemm.zig` into a dynamic library and instructs
   Cargo to link it. Otherwise, the crate uses the built-in Rust fallback implementation.
3. All kernels are feature-gated and opt-in; nothing in the default build requires Zig or native toolchains.

Quick try (you have Zig installed):

```bash
# Run unit tests (builds fallback Rust kernel):
cargo test -p gaba-native-kernels

# Run tests + build the Zig kernel (enable feature):
cargo test -p gaba-native-kernels --features zig

# Run microbenchmarks (this will compare Rust vs native when the feature is enabled):
cargo bench -p gaba-native-kernels --features zig
```

## Benchmarks & reproducibility

- Each kernel crate includes a `benches/` folder using `criterion` so you get detailed, repeatable
  measurements with warmup and statistical reporting.
- We track deterministic fixtures (stored in `crates/*/tests/fixtures`) so search and embedding
  experiments can be replicated exactly.

## Contributing

We welcome small, focused PRs:

- Add a micro-kernel (Zig or Rust) behind a feature flag and include a small benchmark.
- When proposing a change that affects performance, add a criterion benchmark and a short
  benchmarking note in the PR description (machine, OS, CPU model, and any flags used).

If you'd like, open an issue describing the target shape (e.g. GEMM sizes, quantized matmul flavor)
and we can iterate on a hand-tuned Zig kernel together.

## Where to start

1. Read `crates/gaba-native-kernels/README.md` for the kernel contract: inputs are row-major f32.
2. Run `cargo test -p gaba-native-kernels` to validate the fallback implementation.
3. If you have Zig, run `cargo test -p gaba-native-kernels --features zig` then `cargo bench -p gaba-native-kernels --features zig`.

Progress update
- SIMD-optimized Zig GEMM: added compact, comptime-selected inner kernels tuned for NEON (4 lanes) and AVX2 (8 lanes). The kernels keep a safe blocked fallback for portability.
- Bench CSV + baselines: added a CSV bench writer and committed baseline CSVs to `crates/gaba-native-kernels/benches/baseline.csv`.
- CI: fixed the bench workflow and wired `xtask bench_compare` to fail CI on regressions >5%.

</div>
