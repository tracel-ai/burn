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

Measured on 2026-10-23 with an NVIDIA GeForce RTX 3060 (12 GB, device 1), driver
580.178.04, Intel Core i5-12400F and Rust 1.98.0, in release mode. Medians below are
microseconds per solve; lower is better. [All results](benches/solve-rtx3060.csv)
include the 10th and 90th percentile samples. A dash marks an inapplicable variant.

| Dtype | Batch / n / RHS | Auto | Register panel | Blocked RHS | No subgroups | All fallbacks |
|---|---|---:|---:|---:|---:|---:|
| F32 | 1 / 16 / 1 | 73.14 | 97.54 | 81.76 | 81.49 | 114.38 |
| F32 | 256 / 16 / 8 | 196.63 | 138.55 | 226.86 | 224.06 | 272.44 |
| F32 | 64 / 31 / 8 | 223.58 | 195.31 | 207.82 | 252.48 | 246.87 |
| F32 | 64 / 32 / 8 | 208.85 | 181.98 | – | 289.71 | 250.04 |
| F32 | 1 / 64 / 1 | 226.77 | 339.44 | – | 275.44 | 389.02 |
| F32 | 64 / 65 / 8 | 440.51 | 570.72 | – | 741.22 | 870.59 |
| F32 | 1 / 128 / 32 | 718.36 | 935.98 | – | 847.58 | 1027.21 |
| F32 | 16 / 129 / 8 | 604.31 | 815.91 | – | 959.39 | 1040.53 |
| F32 | 1 / 256 / 1 | 1443.08 | 1694.71 | – | 1506.07 | 1874.58 |
| F32 | 16 / 256 / 32 | 1881.77 | 2297.88 | – | 2882.21 | 3299.57 |
| F32 | 1 / 512 / 8 | 2502.96 | 3613.64 | – | 2829.40 | 3982.59 |
| F32 | 1 / 1024 / 8 | 6577.21 | 10017.51 | – | 7351.33 | 10901.40 |
| F64 | 1 / 16 / 1 | 112.97 | 135.91 | 119.45 | – | 142.81 |
| F64 | 256 / 16 / 8 | 636.07 | 409.12 | 834.87 | – | 611.77 |
| F64 | 64 / 31 / 8 | 625.28 | 342.78 | 641.99 | – | 405.15 |
| F64 | 64 / 32 / 8 | 751.43 | 410.21 | – | – | 410.41 |
| F64 | 1 / 64 / 1 | 566.08 | 854.72 | – | – | 857.23 |
| F64 | 64 / 65 / 8 | 2000.16 | 2234.55 | – | – | 2228.66 |
| F64 | 1 / 128 / 32 | 1262.70 | 1745.33 | – | – | 1684.87 |
| F64 | 16 / 129 / 8 | 2084.35 | 2445.20 | – | – | 2435.31 |
| F64 | 1 / 256 / 1 | 2939.48 | 3929.40 | – | – | 3890.49 |
| F64 | 16 / 256 / 32 | 6790.58 | 7672.81 | – | – | 7635.89 |
| F64 | 1 / 512 / 8 | 6420.17 | 9615.12 | – | – | 9636.37 |
| F64 | 1 / 1024 / 8 | 24183.89 | 32754.63 | – | – | 32741.43 |

The combined paths provide up to 1.98x speedup over all fallbacks in this run
(F32, batch 64, n=65, RHS=8). Each specialization has workloads that
benefit, supporting retention of both panel implementations, the small-system RHS kernel,
and subgroup operations. Automatic selection is not uniformly fastest: for F64, batch 64,
n=31 and RHS=8, shared panels take 625.28 us versus 342.78 us
with register panels. The F32 n=31 case also favors blocked RHS substitution. These losses
identify small batched systems as a target for further dispatch tuning across devices.
Other CubeCL runtimes, strided/broadcast performance, low-precision API overhead and fusion
latency were not benchmarked.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
