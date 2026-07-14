# burn-flex Architecture

A pure-Rust CPU backend for [Burn](https://github.com/tracel-ai/burn).

## Goals

From README:

- Fast, memory-efficient CPU backend
- Multi-threading, SIMD, optimized matrix multiplication
- Runs on std, no_std, and WebAssembly
- Supports f16/bf16
- Zero-copy data loading
- Thread-safe by design (Arc-based COW)

## Robustness

burn-flex is tested for edge-case robustness to ensure safe behavior on embedded devices and in
production. This includes:

- **Integer overflow safety**: `wrapping_abs`, `wrapping_neg`, `wrapping_shl/shr` for signed
  integers at type boundaries (e.g. `i64::MIN`), matching PyTorch two's complement semantics
- **Rounding correctness**: Uses `num_traits::Float::round` with a ties-to-even correction,
  correct for the full float range (values beyond integer precision have no fractional bits)
- **Input validation**: Hard assertions for invalid pooling parameters (zero kernel/stride) and
  zero-sized reduce dimensions, preventing undefined behavior on malformed inputs
- **Negative index detection**: Debug assertions on gather/scatter index conversions
- **Index dtype correctness**: Index-producing ops (argmax, argmin, argsort, argwhere,
  sort_with_indices) must respect `out_dtype`/`indices_dtype` parameters. Internally use
  `isize` + `INDEX_DTYPE` for platform portability, then cast to the requested dtype via
  `int_cast` if needed. Never hardcode `i64` for index outputs as it breaks on 32-bit targets.

## Target Platform

**Primary: Apple Silicon M3 (ARM64 + NEON)**

- 128-bit SIMD registers (4x f32, 8x f16)
- Unified memory architecture
- Native f16 support in hardware

**Secondary: x86_64 with AVX2/AVX-512** (via conditional compilation)

---

## Design Principles

1. **Leverage Burn** - Use `burn-backend` types and `burn-std` utilities wherever possible
2. **Portability first** - No platform-specific dependencies; std, no_std, WASM
3. **Zero C dependencies** - Pure Rust only (gemm crate for matrix multiplication)
4. **Simple and direct** - Eager execution, no lazy graphs, no fusion (use `burn-fusion` if needed)
5. **Memory reuse** - Minimize allocations through in-place ops and buffer reuse

---

## Feature Flags

```toml
default = ["std", "simd", "rayon"]
```

| Feature     | Default | Description                                                                 |
| ----------- | ------- | --------------------------------------------------------------------------- |
| `std`       | Yes     | Standard library support                                                    |
| `simd`      | Yes     | Portable SIMD via macerator (enables `macerator`, `aligned-vec`)            |
| `rayon`     | Yes     | Parallel execution for large tensors (forwards `gemm/rayon`)                |
| `x86-v4`    | No      | AVX-512 kernels in gemm for x86_64 (Sapphire Rapids, Zen 4/5, etc.)         |
| `apple-amx` | No      | Apple Silicon AMX matrix coprocessor in gemm (experimental upstream)        |

The `simd` feature also forwards `gemm/wasm-simd128-enable`, a no-op outside WASM.

`gemm` is an always-on required dependency (not behind a feature flag).

### Performance impact on Apple M3 Max (median speedup vs serial baseline)

Measured via `cargo bench -p burn-flex --bench {matmul,attention,conv_ops}` with features
`std,simd` (serial), `std,simd,rayon` (default), and `std,simd,rayon,apple-amx`.

| Workload                                | rayon vs serial | +apple-amx vs rayon | combined |
| --------------------------------------- | --------------- | ------------------- | -------- |
| matmul 1024×1024 f32                    | 7.0x            | 1.7x                | **12.2x** |
| matmul 512×512 f32                      | 3.8x            | 1.5x                | 5.8x     |
| attention self b1·h32·s256·d128         | 1.0x            | 2.0x                | 2.0x     |
| attention self b1·h12·s512·d64          | 1.0x            | 1.6x                | 1.6x     |
| conv2d first_layer 4×3×224×224 k7×7 s2  | 9.8x            | 1.2x                | **11.6x** |
| conv2d large 16×128×64×64 k3×3          | 7.7x            | 1.5x                | 11.1x    |
| conv2d k7×7                             | 6.5x            | 1.4x                | 9.2x     |

Notes:
- Attention ops currently see no rayon uplift; the per-head matmul pipeline does not
  propagate `Parallelism::Rayon` to gemm. AMX still delivers a standalone speedup.
- Small shapes (e.g. `batch8_64x64` matmul, `depthwise_k3_8x32x512` conv1d) can regress
  under rayon due to thread-spawn overhead; a size-based gating in the matmul/conv
  paths would recover those without losing the large-shape wins.
- AMX regresses on transposed operands (`both/rhs_transposed_256x256` matmul drop to
  ~0.55x vs rayon). Avoid `apple-amx` for workloads dominated by transposed GEMM.

---

## Memory Strategy

Minimize allocations wherever possible:

### In-Place Operations

When tensor is contiguous at offset 0, mutate in place:

```rust
fn neg_inplace(mut tensor: FlexTensor) -> FlexTensor {
    if let Some((0, end)) = tensor.layout().contiguous_offsets() {
        let slice: &mut [f32] = tensor.storage_mut();
        for x in slice[..end].iter_mut() {
            *x = -*x;
        }
        tensor
    } else {
        // Allocate new buffer for non-contiguous
        neg_copy(&tensor)
    }
}
```

### Output Buffer Reuse

For binary ops, reuse lhs buffer when contiguous at offset 0:

```rust
fn add(mut lhs: FlexTensor, rhs: &FlexTensor) -> FlexTensor {
    if let Some((0, l_end)) = lhs.layout().contiguous_offsets() {
        if let Some((r_start, r_end)) = rhs.layout().contiguous_offsets() {
            let lhs_storage: &mut [f32] = lhs.storage_mut();
            let rhs_storage: &[f32] = rhs.storage();
            for (l, &r) in lhs_storage[..l_end].iter_mut().zip(&rhs_storage[r_start..r_end]) {
                *l = *l + r;
            }
            return lhs;
        }
    }
    add_alloc(&lhs, rhs)
}
```

### When to Allocate

Only allocate when necessary:

- Shape changes (broadcast, concat, reshape of non-contiguous)
- Non-contiguous input that must become contiguous
- Views/slices with non-zero offset

### Arc-based Copy-on-Write

Tensor storage is wrapped in `Arc<Bytes>` for O(1) cloning and thread-safe COW:

```rust
pub struct FlexTensor {
    data: Arc<Bytes>,  // O(1) clone via refcount increment
    layout: Layout,
    dtype: DType,
}

impl FlexTensor {
    /// Check if this tensor uniquely owns its data
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }

    /// Get mutable access, cloning data if shared (COW)
    pub fn make_data_mut(&mut self) -> &mut Bytes {
        Arc::make_mut(&mut self.data)
    }
}
```

Benefits:

- **O(1) cloning**: `Arc::clone` is just a refcount increment
- **Thread-safe sharing**: `Arc` is `Send + Sync`
- **COW semantics**: `Arc::make_mut` clones only when shared
- **Smart in-place ops**: `is_unique()` enables mutation without allocation

This enables the optimization pattern used throughout:

```rust
fn add_inplace(mut lhs: FlexTensor, rhs: &FlexTensor) -> FlexTensor {
    if lhs.is_unique() && lhs.is_contiguous_at_offset_zero() {
        // Mutate in place - no allocation needed
        let storage = lhs.make_data_mut();
        // ... perform addition ...
        lhs
    } else {
        // Allocate new buffer
        add_alloc(&lhs, rhs)
    }
}
```

Performance impact (vs previous non-Arc implementation):

- Binary ops: **2.6-4.2x faster** than NdArray (was 1.4-1.8x)
- Scalar ops: **2.6x faster** (was 1.8x)
- Memory: 3x less allocation for binary ops (4.2 MB vs 12.6 MB for 1M elements)

---

## Burn Infrastructure We Use

From `burn-backend`:

- `Shape` - tensor dimensions
- `TensorData` - serialized tensor format
- `DType` - runtime dtype enum
- `Element` trait - compile-time element types
- `Backend` trait - the interface we implement
- `*TensorOps` traits - operation interfaces

From `burn-std`:

- `Bytes` - aligned byte storage with COW semantics (our tensor backing store)
- `is_contiguous()` - stride validation
- Platform abstractions for no_std

---

## Core Types

### Layout

Metadata for interpreting storage as an N-dimensional tensor:

```rust
use burn_backend::Shape;

pub struct Layout {
    shape: Shape,
    strides: Vec<isize>,   // Signed strides for zero-copy flip
    start_offset: usize,
}
```

**Signed Strides**

Strides are `isize` (signed) to enable zero-copy flip operations. A negative stride means we iterate
backward through that dimension:

```rust
// Original tensor [1, 2, 3, 4] with shape [4], stride [1], offset 0
// Flipped tensor uses:
//   - offset: 3 (point to last element)
//   - stride: -1 (move backward)
// Iteration: indices 3, 2, 1, 0 -> values 4, 3, 2, 1
```

Many operations are zero-copy (metadata changes only):

- `transpose()` - swap strides
- `narrow()` - adjust offset
- `reshape()` - recompute strides if contiguous
- `broadcast()` - set stride to 0
- `flip()` - negate stride, adjust offset
- `permute()` - reorder strides

**Zero-Copy Flip**

With signed strides, `flip(tensor, axes)` is O(1):

```rust
pub fn flip(&self, axes: &[usize]) -> Self {
    let mut new_strides = self.strides.clone();
    let mut offset_adjustment: isize = 0;

    for &axis in axes {
        let dim_size = self.shape.dims[axis];
        if dim_size > 1 {
            // Move start to the last element in this dimension
            offset_adjustment += (dim_size as isize - 1) * self.strides[axis];
            // Negate stride to iterate backward
            new_strides[axis] = -new_strides[axis];
        }
    }

    let new_start = (self.start_offset as isize + offset_adjustment) as usize;
    Self { shape: self.shape.clone(), strides: new_strides, start_offset: new_start }
}
```

This avoids the O(n) element-by-element copy that would be required with unsigned strides.

### Tensor

Uses `Arc<Bytes>` for O(1) cloning with COW semantics:

```rust
use std::sync::Arc;
use burn_std::Bytes;
use burn_backend::DType;

pub struct FlexTensor {
    data: Arc<Bytes>,  // O(1) clone, COW via Arc::make_mut
    layout: Layout,
    dtype: DType,
}

impl FlexTensor {
    /// Zero-copy typed view of full storage (for use with StridedIter)
    pub fn storage<E: Element + bytemuck::Pod>(&self) -> &[E] {
        bytemuck::cast_slice(&self.data)
    }

    /// Mutable typed view for in-place operations
    pub fn storage_mut<E: Element + bytemuck::Pod>(&mut self) -> &mut [E] {
        bytemuck::cast_slice_mut(&mut self.data)
    }
}
```

Operations dispatch on `dtype` and cast once at the boundary:

```rust
fn add(a: &FlexTensor, b: &FlexTensor) -> FlexTensor {
    match a.dtype {
        DType::F32 => add_impl(a.as_slice::<f32>(), b.as_slice::<f32>()),
        DType::F16 => add_impl(a.as_slice::<f16>(), b.as_slice::<f16>()),
        // ...
    }
}
```

---

## Backend Implementation

```rust
use burn_backend::{Backend, DType};

#[derive(Clone, Copy, Debug, Default)]
pub struct Flex;

impl Backend for Flex {
    type Device = FlexDevice;
    type FloatTensorPrimitive = FlexTensor;
    type IntTensorPrimitive = FlexTensor;
    type BoolTensorPrimitive = FlexTensor;
    type QuantizedTensorPrimitive = FlexQTensor;

    fn name() -> String { "flex".into() }

    fn float_supported_dtypes() -> Vec<DType> {
        vec![DType::F64, DType::F32, DType::F16, DType::BF16]
    }

    fn int_supported_dtypes() -> Vec<DType> {
        vec![DType::I64, DType::I32, DType::I16, DType::I8,
             DType::U64, DType::U32, DType::U16, DType::U8]
    }
}
```

---

## FusionBackend

burn-flex does not implement `FusionBackend`. Without JIT compilation, fusion adds tracking overhead
with no performance benefit. Deferred operations would still execute one-by-one with intermediate
allocations. For CPU with fusion, use `burn-cpu` (which has cubecl's MLIR-based JIT runtime).

---

## Execution Strategy

### Contiguous Fast Path

Most tensors are contiguous. Detect and use direct slice operations:

```rust
fn unary_op<T, F>(storage: &[T], layout: &Layout, f: F) -> Vec<T>
where
    T: Copy,
    F: Fn(T) -> T,
{
    if let Some((start, end)) = layout.contiguous_offsets() {
        storage[start..end].iter().map(|&x| f(x)).collect()
    } else {
        StridedIter::new(layout).map(|i| f(storage[i])).collect()
    }
}
```

### SIMD Kernels

Portable SIMD via macerator, with automatic dispatch per architecture (NEON, AVX2, SSE, WASM
SIMD128) and a scalar fallback module for unsupported platforms:

```rust
use macerator::{Simd, with_simd, vload_unaligned, vstore_unaligned};

#[with_simd]
fn my_kernel<S: Simd>(src: &[f32], dst: &mut [f32]) {
    let lanes = f32::lanes::<S>();
    // load/store vectors, use operator overloading for arithmetic
}

// Dispatch: detects CPU features at runtime
my_kernel(src, dst);
```

The `simd/` module is organized as:

- `portable.rs`: macerator-based binary, comparison, and boolean ops (auto-dispatches to
  NEON/AVX2/SSE/SIMD128/scalar)
- `kernels.rs`: macerator-based reduction kernels (sum, scatter-add)
- `scalar.rs`: fallback for builds without the `simd` feature (bool ops only)
- `aligned.rs`: SIMD-aligned memory allocation

### Parallel Execution

Via rayon for large tensors:

```rust
use rayon::prelude::*;

fn parallel_unary<T, F>(src: &[T], f: F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T) -> T + Send + Sync,
{
    src.par_iter().map(|&x| f(x)).collect()
}
```

### Linear Algebra

gemm crate for matrix multiplication with rayon parallelism:

```rust
use gemm::{gemm, Parallelism};

pub fn matmul_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    let parallelism = if m * n * k >= 192 * 192 * 192 {
        Parallelism::Rayon(0)  // Use all available threads
    } else {
        Parallelism::None
    };

    unsafe {
        gemm(
            m, n, k,
            out.as_mut_ptr(), n as isize, 1,
            1.0,  // alpha
            lhs.as_ptr(), k as isize, 1,
            rhs.as_ptr(), n as isize, 1,
            0.0,  // beta
            parallelism,
        );
    }
}
```

Performance: 1.3-3.4x faster than NdArray (which uses matrixmultiply crate).

### Convolutions (im2col + gemm)

All convolutions use the im2col transformation followed by matrix multiplication. This approach:

- Converts convolution to a well-optimized GEMM operation
- Leverages the same gemm crate used for matmul
- Supports arbitrary strides, padding, dilation, and groups

**Unified 3D Implementation**

Rather than three separate implementations, conv1d and conv2d delegate to conv3d:

```
conv1d([B, C, W], kernel=[K_out, C_in, W_k])
  → expand dims → conv3d([B, C, 1, 1, W], kernel=[K_out, C_in, 1, 1, W_k])
  → squeeze → [B, K_out, W_out]

conv2d([B, C, H, W], kernel=[K_out, C_in, H_k, W_k])
  → expand dims → conv3d([B, C, 1, H, W], kernel=[K_out, C_in, 1, H_k, W_k])
  → squeeze → [B, K_out, H_out, W_out]
```

Size-1 dimensions have negligible overhead since the gemm operation dominates runtime.

**im2col Transformation**

Rearranges input patches into columns for matrix multiplication:

```
Input: [B, C_in, D, H, W]
Kernel: [C_out, C_in/groups, K_d, K_h, K_w]

im2col produces: [spatial_out, C_in/groups * K_d * K_h * K_w]
  where spatial_out = D_out * H_out * W_out

GEMM: W[C_out/groups, col_len] × col[col_len, spatial_out]
  → output[C_out/groups, spatial_out]
```

**Dtype Support**

| Dtype | Implementation                        |
| ----- | ------------------------------------- |
| f32   | Native gemm                           |
| f64   | Native gemm                           |
| f16   | Native gemm (since gemm v0.15)        |
| bf16  | Convert to f32, compute, convert back |

bf16 requires conversion because gemm doesn't have native bf16 support.

**Current Optimizations**

- **Rayon parallelism**: Batches and groups are parallelized via rayon
- **Tiled im2col**: Column buffer is tiled for better cache locality

**Remaining Optimization Opportunities**

1. **Direct convolution**: For small kernels (3x3), direct convolution without im2col can be faster
   due to less memory movement

### Pooling (Unified 3D)

All pooling operations use the same unified 3D pattern as convolutions:

```
pool1d([B, C, W])
  → expand dims → pool3d([B, C, 1, 1, W])
  → squeeze → [B, C, W_out]

pool2d([B, C, H, W])
  → expand dims → pool3d([B, C, 1, H, W])
  → squeeze → [B, C, H_out, W_out]
```

**Supported Operations**

| Operation         | Forward | Backward          |
| ----------------- | ------- | ----------------- |
| max_pool          | Yes     | Yes (via indices) |
| avg_pool          | Yes     | Yes               |
| adaptive_avg_pool | Yes     | Yes               |

**Dtype Support**

| Dtype | Implementation                        |
| ----- | ------------------------------------- |
| f32   | Native                                |
| f64   | Native                                |
| f16   | Native                                |
| bf16  | Convert to f32, compute, convert back |

**Parallelization**

Pooling uses rayon to parallelize over (batch, channel) pairs:

```rust
(0..batch_size).into_par_iter().for_each(|b| {
    (0..channels).into_par_iter().for_each(|c| {
        // Process spatial dimensions for this (b, c) slice
    });
});
```

Each (b, c) slice is independent with good cache locality.

**Max Pool Indices**

Max pool stores flat indices into input spatial dimensions (as i64):

- Used by backward pass to route gradients to correct input positions
- Matches Burn's IntElem type for compatibility

### Conv Transpose (Unified 3D)

Transposed convolutions (deconvolutions) for upsampling. Uses the same unified 3D pattern:

```
conv_transpose1d([B, C_in, W])
  → expand dims → conv_transpose3d([B, C_in, 1, 1, W])
  → squeeze → [B, C_out, W_out]

conv_transpose2d([B, C_in, H, W])
  → expand dims → conv_transpose3d([B, C_in, 1, H, W])
  → squeeze → [B, C_out, H_out, W_out]
```

**Algorithm**

Unlike regular convolution (which gathers input into output), transposed convolution scatters:

```rust
for each input position (id, ih, iw):
    for each kernel position (kd, kh, kw):
        od = id * stride_d + kd * dilation_d - padding_d
        oh = ih * stride_h + kh * dilation_h - padding_h
        ow = iw * stride_w + kw * dilation_w - padding_w
        if (od, oh, ow) in bounds:
            output[od, oh, ow] += input[id, ih, iw] * weight[kd, kh, kw]
```

**Weight Shape**

Conv transpose weight shape is opposite of regular conv:

- Regular conv: `[out_channels, in_channels_per_group, kd, kh, kw]`
- Transpose conv: `[in_channels, out_channels_per_group, kd, kh, kw]`

**Output Size Formula**

```
output_size = (input - 1) * stride + dilation * (kernel - 1) + 1 + padding_out - 2 * padding
```

**Parallelization**

Uses rayon over (batch, output_channel) pairs. For f32, uses atomic adds for thread-safe
accumulation:

```rust
(0..batch_size * out_channels).into_par_iter().for_each(|k| {
    // Scatter input values to output using atomic f32 adds
});
```

**Dtype Support**

| Dtype | Implementation                         |
| ----- | -------------------------------------- |
| f32   | Native with atomic adds                |
| f64   | Native (sequential per output channel) |
| f16   | Native (sequential)                    |
| bf16  | Convert to f32, compute, convert back  |

### Attention (Scaled Dot-Product)

Computes `softmax(Q @ K^T * scale + bias) @ V` with fused scale, softcap, masking (bool + causal),
and additive bias. Auto-selects between two strategies:

**Naive attention** (seq_q * seq_kv <= 256K): Materializes the full [seq_q, seq_kv] score matrix. Per (batch,
head), issues two gemm calls: one for `Q @ K^T` and one for `softmax(scores) @ V`. The softmax loop
applies scale/softcap/mask/bias and normalizes in two passes (find-max, then exp-and-sum). NaN-safe:
fully-masked rows produce zero output, not NaN.

**Flash attention** (seq_q * seq_kv > 256K): Tiles over the KV dimension in chunks of TILE_KV (64 on
native, 32 on WASM). Each tile does a small score gemm, online softmax update (running max/sum with
correction factor to rescale previous tiles), and a value accumulation gemm. Memory is
`O(seq_q * TILE_KV)` per head instead of `O(seq_q * seq_kv)`.

**Why two strategies**: Benchmarks show naive is 5-10% faster for typical transformer shapes
(seq <= 512) because two large gemm calls amortize kernel dispatch overhead better than many small
tiled ones. Flash wins when the score matrix exceeds L2 cache. The threshold is `NAIVE_SCORE_BUDGET`
(256K elements = 1 MB for f32).

Both paths share: gemm via `gemm::gemm`, dtype dispatch with f16/bf16 upcast to f32, scratch buffer
reuse across (batch, head) pairs.

### Unfold (Zero-Copy Strided View)

Unfold extracts sliding windows from a tensor along a dimension. Unlike most backends that copy
data, Flex implements unfold as a **zero-copy strided view**.

**Output Shape**

Given input with shape `[pre..., dim_size, post...]`, unfold along dimension `dim` produces:

- Output shape: `[pre..., windows, post..., window_size]`
- Windows count: `(dim_size - window_size + step) / step`

**Algorithm**

Instead of copying window data, Flex manipulates strides:

```rust
// Build output strides:
// - Dimension `dim` (now windows): input_stride[dim] * step
// - New window_size dimension (appended): input_stride[dim]
// - All other dimensions: same as input

output_strides[dim] = input_strides[dim] * step;  // Windows stride
output_strides.push(input_strides[dim]);          // Within-window stride
```

This makes unfold O(1) regardless of tensor size, simply returning a view with new shape/strides.

**Example**

```
Input: [1, 2, 3, 4, 5] shape [5], stride [1]
Unfold dim=0, size=3, step=1

Output shape: [3, 3] (3 windows of size 3)
Output strides: [1, 1] (window stride = 1*1, within-window stride = 1)

Logical view:
  Window 0: [1, 2, 3]  (offsets 0, 1, 2)
  Window 1: [2, 3, 4]  (offsets 1, 2, 3)
  Window 2: [3, 4, 5]  (offsets 2, 3, 4)
```

**Performance**

| Metric          | Flex                         | NdArray                        |
| --------------- | ---------------------------- | ------------------------------ |
| Time complexity | O(1)                         | O(output_elements)             |
| Memory          | 56-136 bytes (metadata only) | Megabytes (copies all windows) |
| Speedup         | **1,300-156,000x faster**    | -                              |

**Non-Contiguous Output**

The returned tensor is non-contiguous (overlapping windows share storage). Operations that require
contiguous data call `to_contiguous()` internally. Many operations (reduce, matmul, conv) work
directly on strided tensors via `StridedIter`.

### FFT (Real Forward and Inverse)

**Location**: `ops/fft.rs`

Forward (rfft) and inverse (irfft) real FFT via Cooley-Tukey with mixed radix-4/radix-2 DIT.

**Key optimizations:**

- **Complex packing**: For rfft, pack N real values as N/2 complex, run a half-size complex FFT,
  then unpack using Hermitian symmetry. For irfft, reverse the process: repack spectrum, half-size
  inverse FFT, de-interleave. This halves the work compared to a full N-point FFT.
- **Compile-time twiddle tables**: `const fn` Taylor-series sin/cos generates static twiddle factor
  tables for N=2 through 65536. Zero runtime allocation for common sizes. Stored as split f32
  arrays for direct SIMD loads.
- **Unrolled small kernels**: Hardcoded butterfly networks for N=2, 4, 8 with compile-time twiddle
  values (W_4=-i, W_8=sqrt2/2). Eliminates loop overhead for the small inner FFTs produced by
  complex packing.
- **Mixed radix-4/radix-2**: Pairs of radix-2 stages are fused into radix-4 passes, halving the
  number of data passes for better cache behavior. Odd-stage-count FFTs do one radix-2 pass first.
- **SIMD butterflies**: `#[macerator::with_simd]` vectorizes radix-4 butterfly passes across
  consecutive elements within each stage.
- **Inverse via conjugation**: irfft computes IFFT as `(1/N)*conj(FFT(conj(X)))`, reusing the
  forward FFT (with its SIMD path) rather than maintaining a separate inverse kernel.
- **Rayon parallelism**: Batched transforms (multiple independent fibers along the FFT dimension)
  are distributed across threads.

**Dtype support**: f32 (native with SIMD radix-4), f64 (rfft computes in f64 with widened f32
twiddles; irfft truncates to f32 for computation), f16/bf16 (via f32 upcast/downcast).

---

## Optimization Decisions

### Implemented

| Optimization                  | Benefit                             | Notes                                        |
| ----------------------------- | ----------------------------------- | -------------------------------------------- |
| **Arc-based COW**             | O(1) clone, 2.6-4.2x faster ops     | `is_unique()` enables true in-place mutation |
| **Portable SIMD (macerator)** | ~1.5-1.7x for contiguous ops        | Auto-dispatches to NEON/AVX2/SSE/SIMD128     |
| **Rayon parallelism**         | Scales with cores for large tensors | Threshold: 4M elements (memory-bound ops)    |
| **Row-based 2D iteration**    | 5.9x faster for transposed tensors  | Replaces per-element StridedIter             |
| **In-place mutation**         | Eliminates allocation               | When tensor is unique and contiguous         |

### Considered but Skipped

| Optimization                     | Why Skipped                                                                                                                                                                                   |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Cache blocking / loop tiling** | Requires architecture-specific tile sizes. M3 has 128KB L1, but optimal tile size varies by operation, data type, and cache hierarchy. Adds complexity without portable benefit.              |
| **Software prefetching**         | ARM64 `_prefetch` intrinsic is unstable (requires nightly Rust). Apple Silicon has excellent hardware prefetchers that detect strided access patterns automatically. Benefit likely marginal. |
| **Kernel fusion**                | Outside burn-flex scope. Fusion is handled at the Burn framework level via `burn-fusion`. This backend focuses on single-operation efficiency.                                                |
| **Hand-tuned intrinsics**        | Portable SIMD via macerator covers NEON/AVX2/SSE/SIMD128 with a single implementation. Hand-tuned per-arch intrinsics add maintenance burden with marginal benefit for memory-bound ops.      |

### Why Element-wise Ops are Memory-Bound

Element-wise operations (add, mul, etc.) perform ~1 FLOP per 4-8 bytes loaded. Modern CPUs can
execute 100+ FLOPs in the time it takes to load one cache line from RAM. This means:

1. **SIMD helps marginally** - Reduces instruction count but doesn't change memory bandwidth
2. **Avoiding allocation matters more** - In-place mutation eliminates write-allocate traffic
3. **Simple loops auto-vectorize** - Compiler generates good SIMD code for predictable patterns
4. **Hardware prefetchers are effective** - M3 detects sequential and strided patterns automatically

---

## Zero-Copy Loading

`Bytes` from burn-std supports zero-copy scenarios (mmap, external buffers). `FlexTensor` wraps this
in `Arc` for cheap cloning while preserving zero-copy capabilities.

## Thread Safety

`Arc<Bytes>` provides thread-safe sharing with automatic COW:

- `Arc` is `Send + Sync` for safe cross-thread sharing
- `Arc::make_mut` triggers copy only when data is shared
- `Arc::strong_count` enables `is_unique()` checks for in-place optimization
