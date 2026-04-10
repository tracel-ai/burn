# burn-flex

A fast, memory-efficient CPU backend for Burn with multi-threading, SIMD, and optimized matrix
multiplication. Runs on std, no_std, and WebAssembly. Supports f16/bf16, zero-copy data loading, and
is thread-safe by design.

> **[Detailed comparison with burn-ndarray](./COMPARISON.md)**: Full architecture, feature coverage,
> operation-by-operation analysis, and migration path.

### Features

- **Zero-Copy Operations**: Many operations return strided views without copying data:
  - `transpose`, `permute`, `flip`, `narrow`, `slice`
  - `unfold` (sliding windows as strided view instead of materialization)
  - `expand` (broadcast via zero strides)
- **Arc-based Copy-on-Write**: O(1) tensor cloning with automatic COW semantics. In-place mutation
  when uniquely owned.
- **Convolutions**: Unified 3D implementation with im2col + gemm. Conv1d/2d delegate to conv3d.
  Direct conv path for small spatial 1D convs (decomposes into kw gemm calls on NCHW data, skipping
  NHWC conversion and im2col). 1x1 pointwise fast path. Supports groups, dilation, padding.
- **Attention**: Auto-selecting between two gemm-backed strategies based on sequence length. Short
  sequences (score matrix seq_q \* seq_kv <= 256K elements) use naive attention with two large gemm
  calls for lower overhead. Larger shapes use tiled flash attention with online softmax for
  O(TILE_KV) memory per row. Both support causal masking, additive bias (ALiBi), softcap, custom
  scale, and cross-attention.
- **Pooling**: Max pool, avg pool, adaptive avg pool. All via unified 3D with backward pass support.
- **Conv Transpose**: Scatter-based transposed convolutions for upsampling.
- **Portable SIMD**: Uses [macerator](https://crates.io/crates/macerator) for automatic dispatch:
  - aarch64: NEON
  - x86_64: AVX2, AVX512, SSE
  - wasm32: SIMD128
  - Embedded/other: Scalar fallback
- **Matrix Multiplication**: Optimized via [gemm](https://crates.io/crates/gemm) with native f16
  support. Strided batched matmul passes layout strides directly to gemm, so transposed views (e.g.
  `q.matmul(k.swap_dims(-2,-1))`) run at contiguous speed with no copy.
- **FFT**: Forward (rfft) and inverse (irfft) real FFT via Cooley-Tukey with complex packing,
  compile-time twiddle tables, SIMD butterflies, and unrolled small kernels. Works in `no_std`.
- **Parallel Execution**: Optional rayon for large tensors
- **Quantization**: Full quantize/dequantize support with per-tensor and per-block symmetric
  schemes. All ~40 quantized ops (arithmetic, trig, reductions, sorting, etc.) work out of the box.
  Layout ops on quantized tensors (permute, flip, expand, slice, select) are zero-copy. Stores
  scales separately for direct `scale * x_q` dequantization instead of reparsing packed bytes.
- **Dtype Support**: f32, f64, f16 (native), bf16 (via f32 conversion), i8-i64, u8-u64
- **Built on Burn**: Leverages Burn's native infrastructure (`Bytes`, `Shape`, `TensorData`,
  `Element` trait) from burn-backend and burn-std

### Why replace burn-ndarray?

burn-ndarray depends on the [ndarray](https://crates.io/crates/ndarray) crate, which has been slow
to accept contributions and evolve.

burn-flex was built as a from-scratch replacement that addresses the gaps while maintaining full
compatibility with Burn's backend test suite.

### Performance vs burn-ndarray (Apple M3 Max)

#### Compute Performance

burn-ndarray now uses macerator SIMD for f32 elementwise ops
([tracel-ai/burn#2851](https://github.com/tracel-ai/burn/pull/2851)), so contiguous f32 binary/unary
ops are at parity. Flex advantages come from gemm, integer ops (i32 vs i64), structural zero-copy,
and fused kernels.

| Category          | Speedup      | Highlights                              |
| ----------------- | ------------ | --------------------------------------- |
| Binary ops (f32)  | **~1x**      | Both use macerator SIMD for f32         |
| Binary ops (i32)  | **1.8-5.3x** | Flex uses i32, NdArray uses i64         |
| Matmul (square)   | **1.4-3.1x** | gemm at small/large; tied at mid-sizes  |
| Matmul (batched)  | **1.3-2.2x** | Multi-head attention shapes             |
| Matmul (int)      | **3.7-6.5x** | gemm vs matrixmultiply for integers     |
| Conv2d (3x3)      | **1.1-3.7x** | Larger kernels and batches benefit most |
| Conv1d            | **4.3-9.8x** |                                         |
| Conv transpose    | **9.2-84x**  | Direct scatter vs im2col                |
| Attention         | **1.2-3.0x** | Fused softmax, 2-8x lower peak memory   |
| Pooling           | **1.1-3.1x** |                                         |
| Interpolation     | **1.1-6.3x** | Nearest 4-6x, bilinear 1.7-2.8x         |
| Reductions        | **1.3-5.4x** | Near-zero allocation for scalar results |
| Cumulative ops    | **2.1-95x**  | 1D cumsum: 95x faster                   |
| Gather/scatter    | **1.2-6.4x** |                                         |
| Unary (tanh, sin) | **1.3-2.0x** | tanh 2x, sin/cos 1.3-1.5x               |
| Sort              | **2.3-29x**  | 2D sort up to 29x                       |
| Repeat dim        | **8.9-12x**  | Single alloc + memcpy vs N slice_assign |
| Tensor creation   | **16-33x**   | zeros/ones/full                         |
| Embedding         | **4.8-5.8x** |                                         |
| Quantize          | **1.3-1.5x** | Fused 2-pass implementation             |
| FFT (rfft/irfft)  | **yes**      | Native implementation, works in no_std  |

#### Structural Improvements

These reflect better _operation representation_, not faster computation. burn-ndarray eagerly
materializes data for these operations; burn-flex avoids the work entirely through zero-copy views
and separated storage layouts.

| Category      | Improvement      | What changed                                                 |
| ------------- | ---------------- | ------------------------------------------------------------ |
| Dequantize    | **122-238x**     | Direct `scale * x_q` vs reparsing `QuantizedBytes` each call |
| Quantized ops | **6.1-125x**     | Dominated by fast dequantize path above                      |
| Slice/narrow  | **2.1-2,400x**   | Zero-copy strided view vs data copy                          |
| Unfold        | **920-130,000x** | O(1) strided view vs O(n) full materialization               |
| Expand        | **620-2,800x**   | Zero-copy broadcast (stride=0) vs data copy                  |
| Int cast      | **6.3-26,000x**  | Zero-copy reinterpret vs element-wise conversion             |

> **Note on quantization**: burn-ndarray simulates quantization by dequantizing to f32 for most
> operations. The quantized speedups reflect the difference between simulated and native execution,
> not equivalent algorithms running at different speeds.

See [BENCHMARKS.md](./BENCHMARKS.md) for the full breakdown.

### Performance vs candle-core (Apple M3 Max, pure-Rust, no BLAS)

Per-op comparison against [candle-core](https://github.com/huggingface/candle), pure-Rust on both
sides. Across 11 bench files covering every flex op that intersects with candle's CPU API, flex is
as fast or faster on every operation category.

| Category                            | Representative ratio | Notes                            |
| ----------------------------------- | -------------------- | -------------------------------- |
| Batched matmul                      | **8-11x**            | Strided gemm, no copy            |
| Conv1d (wav2vec2)                   | **1.4-7.6x**         | Direct conv path                 |
| Conv2d (ResNet)                     | **1.3-4.0x**         | 1x1 pointwise 4x                 |
| Conv transpose                      | **1.5-1.9x**         |                                  |
| Max/min reductions                  | **3.8-5.1x**         | SIMD + zero-alloc                |
| Pooling (k=3 s=2)                   | **1.8-2.5x**         |                                  |
| Layer norm (fused)                  | **1.6-3.4x**         | Two-pass Welford kernel          |
| Softmax (fused)                     | **1.4-1.7x**         | Three-pass row kernel            |
| Cat, gather, select                 | **1.3-2.5x**         |                                  |
| Nearest2d interpolation             | **1.3-1.4x**         |                                  |
| Elementwise, matmul, gelu, view ops | tied                 | Both at memory bandwidth ceiling |

### Status

- All `burn-backend-tests` pass across all feature flag combinations:
  - `no-default-features` (no_std, no SIMD, no rayon)
  - `no-default-features + simd` (no_std with SIMD)
  - `std`
  - `std + simd`
  - `std + rayon`
  - `std + simd + rayon` (default)
- Burn's `burn-no-std-tests` integration suite passes (MNIST model inference in `#![no_std]`)
- Builds for embedded and WebAssembly targets:
  - `thumbv6m-none-eabi` (ARM Cortex-M0+, no atomic pointers)
  - `thumbv7m-none-eabi` (ARM Cortex-M3)
  - `wasm32-unknown-unknown`
- Passes [Miri](https://github.com/rust-lang/miri) (undefined behavior detector) on all burn-flex
  code (quantization tests skipped due to upstream UB in burn-std). Validates memory safety of
  unsafe pointer arithmetic, bytemuck casts, and Send/Sync implementations.
- Tested for edge-case robustness: integer overflow at type boundaries, large-float rounding,
  invalid pooling parameters, zero-sized dimensions. Safe for embedded devices.
- All ONNX model checks in `burn-onnx` pass
- Real model inference verified:
  - [ALBERT](https://huggingface.co/albert/albert-base-v2) (masked language model, all v2 variants)
  - [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (sentence embeddings, L6
    and L12)

### Documentation

- [COMPARISON.md](./COMPARISON.md) - Comprehensive comparison with burn-ndarray
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Design decisions, memory strategy, and implementation
  patterns
- [BENCHMARKS.md](./BENCHMARKS.md) - Full benchmark results (Flex vs NdArray)
- [ACKNOWLEDGMENTS.md](./ACKNOWLEDGMENTS.md) - Projects that influenced burn-flex
