# Acknowledgments

burn-flex draws on ideas and techniques from several open-source projects.

## ndarray

[ndarray](https://github.com/rust-ndarray/ndarray) - N-dimensional array library for Rust.

- **8-fold unrolled reduction loop**: Our `sum_f32` implementation in `simd/kernels.rs` uses
  ndarray's `unrolled_fold` pattern, where eight independent accumulators allow LLVM to emit optimal
  SIMD code without explicit intrinsic dispatch.

## Candle

[Candle](https://github.com/huggingface/candle) - Minimalist ML framework by Hugging Face.

- **Tiled im2col for convolutions**: Our convolution implementation uses a tiled im2col approach
  (TILE_SIZE=512) inspired by Candle, processing output in fixed-size tiles for better L2 cache
  utilization and enabling tile-level parallelism.

## gemm / macerator

[gemm](https://github.com/sarah-ek/gemm) and [macerator](https://github.com/wingertge/macerator),
part of the [faer](https://github.com/sarah-ek/faer-rs) ecosystem.

- **gemm**: Powers all matrix multiplication and convolution GEMM calls (matmul, conv im2col,
  deformable conv). Provides strided memory access so we can multiply transposed tensors without
  copying, and native f16 support.
- **macerator**: Provides portable SIMD dispatch for scatter-add reductions across NEON, AVX2, and
  WASM SIMD128 with a scalar fallback.
