# Burn Linear Algebra

> Linear algebra operations for [Burn](https://github.com/tracel-ai/burn) tensors

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-linalg.svg)](https://crates.io/crates/burn-linalg)
[![Documentation](https://docs.rs/burn-linalg/badge.svg)](https://docs.rs/burn-linalg)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

- Decompositions: `lu`, `qr` and `svd`.
- Linear systems: `solve` for square, nonsingular matrices with vector or matrix right-hand sides.
- `det`, `trace`, `diag`, `outer`, `matvec` and `cosine_similarity`.
- Vector and matrix norms: L0, L1, L2, Lp, max and min absolute value, and `vector_normalize`.

## Usage

Enable both `linalg` and a backend feature on `burn`, and use the operations through
`burn::linalg`:

```toml
burn = { version = "0.22", features = ["linalg", "flex"] }
```

No execution backend is enabled by default when depending on this crate directly; select `flex`,
`wgpu` or another backend feature on it. Enabling only `burn/flex` does not enable this crate.

Linear algebra previously lived at `burn_tensor::linalg`; that path is removed in 0.22.

## CUDA solve performance

The ignored `solve_cubecl::benchmarks::cuda::compare_solve_paths` test compares the
private device solver with its fallback paths. Run it on an otherwise idle GPU:

```sh
BURN_SOLVE_BENCH_DEVICE=1 cargo test -p burn-linalg --release \
  --no-default-features --features std,cuda --lib compare_solve_paths -- \
  --ignored --nocapture
```

The device index defaults to zero. Inputs are contiguous, row-permuted, diagonally dominant
matrices with known solutions. Every variant is checked against that solution before timing.
After five warmups per variant, 40 samples are interleaved with rotating variant order.
Wall-clock timings include allocation, kernel launches, singularity readback and device
completion; input generation/transfers and JIT compilation are excluded. These are direct
`CubeBackend` solve timings, without public API rank promotion, low-precision casts or fusion.

The comparison variants are:

- `register_panel`: use the register-backed factorization fallback instead of shared-memory
  panels, including the fallback's fixed panel width.
- `blocked_rhs`: use the blocked triangular solve instead of the small-system shared-memory
  RHS kernel. This comparison applies only when `n < 32`.
- `no_subgroups`: use shared-memory pivot reductions and RHS updates instead of subgroup
  reductions/shuffles. Subgroup paths currently apply to F32 only.
- `fallback`: disable all three optimizations together.

Measured on 2026-10-24 with an NVIDIA GeForce RTX 3060 (12 GB, device 1), driver
580.178.04, Intel Core i5-12400F and Rust 1.98.0, in release mode, after merging main
at `3da5d2a3c` (CubeCL `81fc3092`, CubeK `ed2b87d6`). Medians below are microseconds
per solve; lower is better. [All results](benches/solve-rtx3060.csv) include the 10th and
90th percentile samples. A dash marks an inapplicable variant.

| Dtype | Batch / n / RHS | Auto | Register panel | Blocked RHS | No subgroups | All fallbacks |
|---|---|---:|---:|---:|---:|---:|
| F32 | 1 / 16 / 1 | 80.20 | 98.16 | 83.43 | 81.28 | 113.13 |
| F32 | 256 / 16 / 8 | 202.88 | 141.86 | 237.56 | 230.23 | 275.83 |
| F32 | 64 / 31 / 8 | 228.77 | 199.61 | 221.79 | 259.56 | 248.54 |
| F32 | 64 / 32 / 8 | 221.07 | 192.56 | – | 292.38 | 251.44 |
| F32 | 1 / 64 / 1 | 229.83 | 325.51 | – | 270.58 | 366.75 |
| F32 | 64 / 65 / 8 | 426.92 | 540.57 | – | 592.11 | 690.48 |
| F32 | 1 / 128 / 32 | 481.34 | 663.96 | – | 565.12 | 754.36 |
| F32 | 16 / 129 / 8 | 589.97 | 809.65 | – | 766.82 | 997.17 |
| F32 | 1 / 256 / 1 | 1025.59 | 1448.66 | – | 1176.83 | 1613.71 |
| F32 | 16 / 256 / 32 | 1879.20 | 2263.27 | – | 2751.90 | 3157.41 |
| F32 | 1 / 512 / 8 | 2221.07 | 3298.88 | – | 2472.11 | 3613.23 |
| F32 | 1 / 1024 / 8 | 6409.92 | 9907.14 | – | 6907.90 | 10469.73 |
| F64 | 1 / 16 / 1 | 108.87 | 132.11 | 113.17 | – | 137.02 |
| F64 | 256 / 16 / 8 | 597.12 | 390.45 | 801.55 | – | 597.01 |
| F64 | 64 / 31 / 8 | 506.64 | 340.15 | 556.94 | – | 390.69 |
| F64 | 64 / 32 / 8 | 566.07 | 388.90 | – | – | 388.88 |
| F64 | 1 / 64 / 1 | 401.35 | 606.16 | – | – | 605.83 |
| F64 | 64 / 65 / 8 | 1725.20 | 1939.28 | – | – | 1932.71 |
| F64 | 1 / 128 / 32 | 1112.23 | 1442.82 | – | – | 1441.70 |
| F64 | 16 / 129 / 8 | 1676.53 | 2067.69 | – | – | 2067.10 |
| F64 | 1 / 256 / 1 | 2617.32 | 3510.81 | – | – | 3510.61 |
| F64 | 16 / 256 / 32 | 6721.35 | 7613.85 | – | – | 7622.48 |
| F64 | 1 / 512 / 8 | 6209.58 | 9356.10 | – | – | 9361.92 |
| F64 | 1 / 1024 / 8 | 23837.97 | 31688.44 | – | – | 31696.15 |

The combined paths provide up to 1.69x speedup over all fallbacks in this run
(F32, batch 16, n=129, RHS=8). Each specialization has workloads that
benefit, supporting retention of both panel implementations, the small-system RHS kernel,
and subgroup operations. Automatic selection is not uniformly fastest: for
F64, batch 256, n=16 and RHS=8, Auto takes 597.12 us
versus 390.45 us with `register_panel`. These losses identify dispatch decisions for
further tuning across devices.
Other CubeCL runtimes, strided/broadcast performance, low-precision API overhead and fusion
latency were not benchmarked.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
