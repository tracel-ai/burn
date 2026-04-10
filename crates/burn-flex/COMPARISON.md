# burn-flex vs burn-ndarray: Comprehensive Comparison

This document compares burn-flex (proposed replacement) against burn-ndarray (current CPU backend)
to demonstrate full coverage and the architectural differences between the two.

## Executive Summary

burn-flex is a from-scratch CPU backend built to replace burn-ndarray. The
[ndarray](https://crates.io/crates/ndarray) crate has been slow to evolve: it lacks f16/bf16
support, is limited to 6 dimensions, uses unsigned-only strides (preventing zero-copy flip), and
simulates quantization rather than executing natively. burn-flex addresses all of these while
passing the full `burn-backend-tests` suite, all ONNX model checks, and real model inference
(ALBERT, MiniLM).

Performance improvements fall into two categories:

- **Compute gains** (1.1-9.7x): Better algorithms and libraries (gemm over matrixmultiply, Arc COW
  for buffer reuse, SIMD reductions).
- **Structural improvements** (up to 166,000x): Operations that burn-ndarray eagerly materializes
  (unfold, expand, slice, dequantize) are represented as zero-copy views or direct lookups in
  burn-flex, avoiding the work entirely.

burn-flex uses significantly less memory, supports f16/bf16 natively, runs on no_std/WASM/embedded,
and has no dimension limit.

---

## 1. Architecture

### Tensor Representation

| Aspect      | burn-flex                                             | burn-ndarray                                                              |
| ----------- | ----------------------------------------------------- | ------------------------------------------------------------------------- |
| Storage     | `Arc<Bytes>` (type-erased bytes)                      | `enum NdArrayTensor { F64(NdArrayStorage<f64>), F32(...), ... }`          |
| Dtype       | Runtime `DType` field on `FlexTensor`                 | Compile-time via enum variant                                             |
| Dispatch    | `match dtype` at op entry, cast once                  | `execute_with_dtype!` macro expands match for every op                    |
| Clone cost  | O(1) Arc refcount increment                           | O(1) ArcArray refcount increment                                          |
| COW         | `Arc::make_mut` / `is_unique()`                       | `ArcArray::is_unique()` + `NdArrayStorage::Borrowed` always returns false |
| Metadata    | `Layout { shape, strides: Vec<isize>, start_offset }` | ndarray's internal strides (`usize` only)                                 |
| Stride sign | **Signed** (`isize`) for zero-copy flip               | **Unsigned** (`usize`), flip requires data copy                           |

**FlexTensor** (44 bytes without shape vec):

```rust
struct FlexTensor {
    data: Arc<Bytes>,    // 8 bytes (pointer)
    layout: Layout,      // shape + strides + offset
    dtype: DType,        // 1 byte enum
}
```

**NdArrayTensor** (enum with 11 typed variants):

```rust
enum NdArrayTensor {
    F64(NdArrayStorage<f64>),
    F32(NdArrayStorage<f32>),
    // ... 9 more variants
}
```

**Key insight**: Flex uses one struct for all dtypes with runtime dispatch. NdArray uses a typed
enum with macro-based dispatch. Flex's approach is simpler (no macros, no generics plumbing) and
enables operations to handle all dtypes uniformly.

### Backend Type

| Aspect        | burn-flex                    | burn-ndarray                                            |
| ------------- | ---------------------------- | ------------------------------------------------------- |
| Type          | `struct Flex;` (unit struct) | `struct NdArray<E=f32, I=i64, Q=i8>` (3 generic params) |
| Float element | Runtime (f32/f64/f16/bf16)   | Compile-time `E: FloatNdArrayElement` (f32 or f64 only) |
| Int element   | Runtime (i8-i64, u8-u64)     | Compile-time `I: IntNdArrayElement`                     |
| Quant element | Runtime                      | Compile-time `Q: QuantElement`                          |

Flex eliminates generic parameters entirely. Users write `Flex` instead of `NdArray<f32, i64, i8>`.
Dtype selection happens at runtime via `DType`.

---

## 2. Feature Coverage

### Float Dtypes

| Dtype    | burn-flex                                                   | burn-ndarray          |
| -------- | ----------------------------------------------------------- | --------------------- |
| f32      | Full support (native)                                       | Full support (native) |
| f64      | Full support (native)                                       | Full support (native) |
| **f16**  | **Full support (native)**                                   | **Not supported**     |
| **bf16** | **Full support (via f32 conversion for compute-heavy ops)** | **Not supported**     |
| Flex32   | Not applicable                                              | Maps to f32           |

burn-flex's f16 support is native for all operations. For matmul and convolution, the gemm crate has
native f16 kernels (since v0.15). bf16 converts to f32 for compute-heavy ops (matmul, conv) because
gemm lacks native bf16 support.

### Integer Dtypes

| Dtype | burn-flex    | burn-ndarray |
| ----- | ------------ | ------------ |
| i64   | Full support | Full support |
| i32   | Full support | Full support |
| i16   | Full support | Full support |
| i8    | Full support | Full support |
| u64   | Full support | Full support |
| u32   | Full support | Full support |
| u16   | Full support | Full support |
| u8    | Full support | Full support |

Both backends support the same integer dtypes.

### Bool

| Feature    | burn-flex                 | burn-ndarray                            |
| ---------- | ------------------------- | --------------------------------------- |
| Storage    | `u8` (1 byte per element) | `bool` (1 byte per element via ndarray) |
| Operations | All BoolTensorOps         | All BoolTensorOps                       |

### Quantization

| Feature        | burn-flex                                                       | burn-ndarray                            |
| -------------- | --------------------------------------------------------------- | --------------------------------------- |
| Quantize       | Per-tensor and per-block symmetric                              | Per-tensor and per-block symmetric      |
| Dequantize     | `scale * x_q` (direct multiply, **135-232x faster**)            | Reparses `QuantizedBytes` on every call |
| Scale storage  | `Vec<f32>` stored separately                                    | `QParams<f32>` in `NdArrayQTensor`      |
| Q layout ops   | **Zero-copy** (permute, flip, expand, slice, select)            | Copies entire tensor                    |
| Q ordering ops | **Skip dequantization** (argmax, argmin, gather on i8 directly) | Dequantize to f32, then operate         |
| QuantStore     | Native                                                          | Native                                  |
| QuantValue     | Q8F, Q8S                                                        | Q8F, Q8S (+ Q4/Q2 for export_tests)     |

The fundamental difference is scale storage. Flex stores scales separately so dequantization is a
simple `scale * x_q` multiply. NdArray stores everything in `QuantizedBytes` which must be parsed on
every access, making it the bottleneck for all quantized operations.

---

## 3. Operation Coverage

### Tensor Operations (FloatTensorOps)

All operations listed below are implemented by both backends unless marked otherwise.

| Operation                                                         | burn-flex | burn-ndarray | Notes                                                 |
| ----------------------------------------------------------------- | --------- | ------------ | ----------------------------------------------------- |
| from_data                                                         | Yes       | Yes          |                                                       |
| into_data                                                         | Yes       | Yes          |                                                       |
| random                                                            | Yes       | Yes          |                                                       |
| empty/zeros/ones                                                  | Yes       | Yes          |                                                       |
| full                                                              | Yes       | Yes          |                                                       |
| add / sub / mul / div                                             | Yes       | Yes          |                                                       |
| add_scalar / sub_scalar / mul_scalar / div_scalar                 | Yes       | Yes          |                                                       |
| remainder                                                         | Yes       | Yes          |                                                       |
| remainder_scalar                                                  | Yes       | Yes          |                                                       |
| matmul                                                            | Yes       | Yes          | Flex uses gemm, NdArray uses matrixmultiply           |
| neg                                                               | Yes       | Yes          |                                                       |
| recip                                                             | Yes       | Yes          |                                                       |
| swap_dims / permute                                               | Yes       | Yes          | Both zero-copy                                        |
| reshape                                                           | Yes       | Yes          | Both zero-copy when contiguous                        |
| gather / scatter_add                                              | Yes       | Yes          |                                                       |
| select / select_add                                               | Yes       | Yes          |                                                       |
| slice / slice_assign                                              | Yes       | Yes          | Flex: zero-copy view; NdArray: may copy               |
| mask_fill / mask_where                                            | Yes       | Yes          |                                                       |
| equal / not_equal / greater / lower / greater_equal / lower_equal | Yes       | Yes          |                                                       |
| equal_elem / not_equal_elem / greater_elem / lower_elem           | Yes       | Yes          |                                                       |
| sum / sum_dim / mean / mean_dim / prod / prod_dim                 | Yes       | Yes          |                                                       |
| max / max_dim / max_dim_with_indices                              | Yes       | Yes          |                                                       |
| min / min_dim / min_dim_with_indices                              | Yes       | Yes          |                                                       |
| argmax / argmin                                                   | Yes       | Yes          |                                                       |
| any / any_dim / all / all_dim                                     | Yes       | Yes          |                                                       |
| exp / log / log1p                                                 | Yes       | Yes          |                                                       |
| powf / powf_scalar / powi / powi_scalar                           | Yes       | Yes          |                                                       |
| sqrt / abs / sign                                                 | Yes       | Yes          |                                                       |
| cos / sin / tanh                                                  | Yes       | Yes          |                                                       |
| erf                                                               | Yes       | Yes          |                                                       |
| cat                                                               | Yes       | Yes          |                                                       |
| into_int / into_bool                                              | Yes       | Yes          |                                                       |
| clamp / clamp_min / clamp_max                                     | Yes       | Yes          |                                                       |
| expand                                                            | Yes       | Yes          | Flex: zero-copy; NdArray: copies                      |
| flip                                                              | Yes       | Yes          | Flex: zero-copy (signed strides); NdArray: copies     |
| repeat_dim                                                        | Yes       | Yes          |                                                       |
| sort / sort_with_indices / argsort                                | Yes       | Yes          |                                                       |
| cumsum / cumprod / cummin / cummax                                | Yes       | Yes          |                                                       |
| narrow                                                            | Yes       | Yes          | Flex: zero-copy; NdArray: may copy                    |
| chunk                                                             | Yes       | Yes          |                                                       |
| cross                                                             | Yes       | Yes          |                                                       |
| unfold                                                            | Yes       | Yes          | Flex: zero-copy (strided view); NdArray: materializes |
| round / floor / ceil                                              | Yes       | Yes          |                                                       |
| cast                                                              | Yes       | Yes          |                                                       |
| grid_sample_2d                                                    | Yes       | Yes          |                                                       |
| bool_select                                                       | Yes       | Yes          |                                                       |
| int_powi                                                          | Yes       | Yes          |                                                       |

### Module Operations (ModuleOps)

| Operation                        | burn-flex | burn-ndarray | Notes                                                                                  |
| -------------------------------- | --------- | ------------ | -------------------------------------------------------------------------------------- |
| conv1d                           | Yes       | Yes          | Flex: delegates to conv3d                                                              |
| conv2d                           | Yes       | Yes          | Flex: delegates to conv3d                                                              |
| conv3d                           | Yes       | Yes          | Flex: unified implementation                                                           |
| conv_transpose1d                 | Yes       | Yes          | Flex: delegates to conv_transpose3d                                                    |
| conv_transpose2d                 | Yes       | Yes          | Flex: delegates to conv_transpose3d                                                    |
| conv_transpose3d                 | Yes       | Yes          | Flex: unified implementation                                                           |
| deform_conv2d                    | Yes       | Yes          |                                                                                        |
| deform_conv2d_backward           | Yes       | Yes          |                                                                                        |
| avg_pool2d                       | Yes       | Yes          | Flex: delegates to pool3d                                                              |
| avg_pool2d_backward              | Yes       | Yes          |                                                                                        |
| max_pool2d                       | Yes       | Yes          | Flex: delegates to pool3d                                                              |
| max_pool2d_with_indices          | Yes       | Yes          |                                                                                        |
| max_pool2d_with_indices_backward | Yes       | Yes          |                                                                                        |
| adaptive_avg_pool2d              | Yes       | Yes          |                                                                                        |
| adaptive_avg_pool2d_backward     | Yes       | Yes          |                                                                                        |
| interpolate                      | Yes       | Yes          | Nearest, bilinear, bicubic                                                             |
| attention (SDPA)                 | Yes       | Yes          | Flex: auto-selects naive or flash by score matrix size; NdArray: matmul + softmax      |
| rfft                             | Yes       | No           | Flex: Cooley-Tukey with complex packing, radix-4, SIMD, compile-time twiddles. no_std. |
| irfft                            | Yes       | No           | Flex: Inverse packing trick, SIMD via conjugate-forward-conjugate. no_std.             |

### Int and Bool Operations

Both backends implement all IntTensorOps and BoolTensorOps. The operations mirror float ops where
applicable (arithmetic, comparison, reduction, gather/scatter, slice, etc.) plus type-specific
operations (int_random uniform, bool_not, bool_and, bool_or, bool_xor).

### Quantized Operations (QTensorOps)

Both backends implement all QTensorOps. The ops follow a dequantize-op-requantize pattern for most
operations. Flex optimizes by:

- Storing scales separately for O(1) dequantization access
- Zero-copy layout ops on quantized tensors (permute, flip, expand, slice, select)
- Skipping dequantization for ordering ops (argmax, argmin, gather with tensor-level quant)

### Activation Operations (ActivationOps)

Both backends implement all ActivationOps via the default trait implementations (relu, gelu, etc.).

### Transaction Operations

Both backends implement TransactionOps for batched tensor operations.

---

## 4. Dimension Limits

| Aspect         | burn-flex                        | burn-ndarray                                   |
| -------------- | -------------------------------- | ---------------------------------------------- |
| Max dimensions | **Unlimited** (arbitrary rank)   | **6** (hardcoded in reshape macro)             |
| Enforcement    | Dynamic `Vec<isize>` for strides | Static `Dim<[usize; N]>` requires match on 1-6 |

burn-ndarray's dimension limit comes from its `reshape!` macro which matches on dimensions 1-6:

```rust
match $D {
    1 => reshape!(ty $ty, n 1, ...),
    // ...
    6 => reshape!(ty $ty, n 6, ...),
    _ => panic!("NdArray supports arrays up to 6 dimensions"),
}
```

burn-flex uses `IxDyn`-equivalent dynamic shapes with no upper bound.

---

## 5. Zero-Copy Operations

| Operation      | burn-flex                         | burn-ndarray                   |
| -------------- | --------------------------------- | ------------------------------ |
| transpose      | Zero-copy (swap strides)          | Zero-copy (ndarray view)       |
| permute        | Zero-copy (reorder strides)       | Zero-copy (ndarray view)       |
| reshape        | Zero-copy if contiguous           | Zero-copy if standard layout   |
| slice / narrow | Zero-copy (offset + strides)      | May allocate depending on path |
| **flip**       | **Zero-copy (negate stride)**     | **Copies data**                |
| **unfold**     | **Zero-copy (O(1) strided view)** | **O(n) full materialization**  |
| **expand**     | **Zero-copy (set stride to 0)**   | **Copies data**                |

Flex's signed strides (`isize`) enable zero-copy flip, which is impossible with ndarray's unsigned
strides. The unfold operation is especially dramatic: Flex returns a strided view in ~50ns
regardless of size, while NdArray copies all window data (milliseconds for large tensors).

---

## 6. Memory Strategy

### In-Place Mutation

| Strategy           | burn-flex                                             | burn-ndarray                       |
| ------------------ | ----------------------------------------------------- | ---------------------------------- |
| Unique check       | `Arc::strong_count(&data) == 1`                       | `ArcArray::is_unique()`            |
| In-place threshold | Contiguous at offset 0 AND unique                     | Unique (via SIMD ops, not all ops) |
| Binary op reuse    | Reuses lhs buffer when contiguous                     | Allocates new for most ops         |
| Allocation savings | 3x less for binary ops (4.2 MB vs 12.6 MB for 1M f32) | Standard ndarray allocation        |

### Zero-Copy Loading

Both backends support zero-copy loading from external sources (burnpack files, mmap'd data):

| Feature     | burn-flex                                 | burn-ndarray                                     |
| ----------- | ----------------------------------------- | ------------------------------------------------ |
| Mechanism   | `Arc<Bytes>` wraps borrowed data directly | `NdArrayStorage::Borrowed` holds `Bytes` + shape |
| COW trigger | `Arc::make_mut` clones on shared mutation | `into_owned()` copies borrowed to ArcArray       |
| View access | `storage::<E>()` via bytemuck cast        | `view()` via unsafe ArrayView from raw pointer   |

---

## 7. SIMD

| Aspect       | burn-flex                                                   | burn-ndarray                                   |
| ------------ | ----------------------------------------------------------- | ---------------------------------------------- |
| Library      | macerator (required with `simd` feature)                    | macerator (optional with `simd` feature)       |
| Dispatch     | `Arch::new().dispatch(kernel)`                              | Same macerator dispatch                        |
| ISAs         | NEON, AVX2, AVX512, SSE, SIMD128, scalar fallback           | NEON, AVX2, SSE, SIMD128, scalar fallback      |
| Coverage     | Binary ops, comparisons, boolean ops, reductions, unary ops | Binary ops, comparisons, unary ops, conv, pool |
| Without SIMD | Scalar fallback module (`simd/scalar.rs`)                   | Falls back to ndarray operations               |

Both use macerator for portable SIMD. NdArray additionally has SIMD-optimized conv and pool kernels.
Flex relies on the gemm crate's built-in SIMD for matmul/conv performance.

---

## 8. Matrix Multiplication

| Aspect      | burn-flex                              | burn-ndarray                                         |
| ----------- | -------------------------------------- | ---------------------------------------------------- |
| Library     | `gemm` crate (v0.18)                   | `matrixmultiply` crate (via ndarray)                 |
| f32         | Native gemm kernel                     | matrixmultiply                                       |
| f64         | Native gemm kernel                     | matrixmultiply                                       |
| f16         | **Native gemm kernel (since v0.15)**   | **Not supported**                                    |
| bf16        | Convert to f32, gemm, convert back     | **Not supported**                                    |
| i32 matmul  | Manual nested loop                     | Manual nested loop                                   |
| Parallelism | Rayon via gemm (threshold: 192^3)      | Rayon via iter_range_par macro                       |
| Batched     | Parallel over batches + per-batch gemm | Parallel over batches + ndarray general_mat_mul      |
| Broadcast   | Handles batch broadcast natively       | Handles batch broadcast via stride mapping           |
| BLAS option | No (pure Rust only)                    | Yes (Accelerate, OpenBLAS, Netlib via feature flags) |

burn-ndarray offers optional BLAS acceleration (Accelerate on macOS, OpenBLAS, Netlib) through
feature flags. burn-flex uses only the gemm crate, which is pure Rust but highly optimized with its
own SIMD kernels. The gemm crate consistently outperforms matrixmultiply by 1.3-3.4x on Apple M3
Max.

---

## 9. Convolutions

| Aspect       | burn-flex                     | burn-ndarray                                       |
| ------------ | ----------------------------- | -------------------------------------------------- |
| Algorithm    | im2col + gemm (unified 3D)    | Direct computation (per-dimension implementations) |
| conv1d       | Delegates to conv3d           | Separate implementation                            |
| conv2d       | Delegates to conv3d           | Separate implementation                            |
| conv3d       | Single unified implementation | Separate implementation                            |
| f16 support  | **Native gemm**               | **Not supported**                                  |
| bf16 support | **Via f32 conversion**        | **Not supported**                                  |
| Parallelism  | Rayon over batches and groups | iter_range_par over batches                        |
| SIMD conv    | Via gemm SIMD kernels         | macerator-based SIMD conv kernel                   |

Flex's unified 3D approach means one implementation covers all dimensionalities. The tradeoff is
that 1D/2D convolutions expand dimensions (negligible overhead since gemm dominates).

NdArray has dedicated SIMD conv/pool kernels via macerator, which can be faster for specific
patterns. Flex relies on the gemm crate's SIMD for all compute-heavy paths.

---

## 10. Parallelism

| Aspect           | burn-flex                             | burn-ndarray                              |
| ---------------- | ------------------------------------- | ----------------------------------------- |
| Library          | rayon (optional)                      | rayon (optional, called "multi-threads")  |
| Feature flag     | `rayon`                               | `multi-threads`                           |
| Threshold        | 4M elements for memory-bound ops      | Via `run_par!` / `iter_range_par!` macros |
| Scope            | Large tensors, batch dims, pool, conv | Matmul batches, ops via macros            |
| gemm parallelism | Rayon via `Parallelism::Rayon(0)`     | matrixmultiply threading                  |
| Without feature  | Single-threaded (all ops work)        | Single-threaded (all ops work)            |

---

## 11. Platform Support

| Target                          | burn-flex                          | burn-ndarray              |
| ------------------------------- | ---------------------------------- | ------------------------- |
| x86_64 (std)                    | Yes                                | Yes                       |
| aarch64 (std)                   | Yes (primary target)               | Yes                       |
| wasm32-unknown-unknown          | **Yes (verified)**                 | Yes (claimed, categories) |
| thumbv6m-none-eabi (Cortex-M0+) | **Yes (verified, no atomic ptrs)** | Not verified              |
| thumbv7m-none-eabi (Cortex-M3)  | **Yes (verified)**                 | Not verified              |
| no_std                          | **Yes (tested, MNIST inference)**  | Yes (supported)           |

burn-flex has been explicitly tested on embedded targets with Burn's `burn-no-std-tests` integration
suite (MNIST model inference).

---

## 12. Dependencies

### burn-flex

| Dependency   | Purpose                             | Required           |
| ------------ | ----------------------------------- | ------------------ |
| burn-backend | Backend traits, types               | Always             |
| burn-ir      | BackendIr trait                     | Always             |
| burn-std     | Bytes, Shape, platform abstractions | Always             |
| half         | f16/bf16 types                      | Always             |
| bytemuck     | Zero-copy type casting              | Always             |
| num-traits   | Numeric traits (libm for no_std)    | Always             |
| gemm         | Matrix multiplication               | Always             |
| macerator    | Portable SIMD                       | Optional (`simd`)  |
| aligned-vec  | SIMD-aligned allocation             | Optional (`simd`)  |
| rayon        | Parallelism                         | Optional (`rayon`) |

**Total: 7 required + 3 optional**

### burn-ndarray

| Dependency           | Purpose                          | Required                   |
| -------------------- | -------------------------------- | -------------------------- |
| burn-backend         | Backend traits, types            | Always                     |
| burn-std             | Platform abstractions            | Always                     |
| burn-autodiff        | Autodiff support                 | Optional (`std`)           |
| burn-ir              | IR types                         | Always                     |
| ndarray              | N-dimensional array library      | Always                     |
| matrixmultiply       | Matrix multiplication            | Always                     |
| atomic_float         | Atomic f32/f64                   | Always                     |
| const-random         | Compile-time random              | Always                     |
| libm                 | Math functions for no_std        | Always                     |
| num-traits           | Numeric traits                   | Always                     |
| paste                | Macro utilities                  | Always                     |
| rand                 | Random number generation         | Always                     |
| macerator            | Portable SIMD                    | Optional (`simd`)          |
| bytemuck             | Type casting                     | Optional (`simd`)          |
| itertools            | Iterator utilities               | Optional (`simd`)          |
| seq-macro            | Sequence macros                  | Optional (`simd`)          |
| rayon                | Parallelism                      | Optional (`multi-threads`) |
| blas-src             | BLAS bindings                    | Optional (`blas-*`)        |
| openblas-src         | OpenBLAS                         | Optional (`blas-openblas`) |
| portable-atomic      | Atomic for no-atomic-ptr targets | Conditional                |
| portable-atomic-util | Atomic utilities                 | Conditional                |

**Total: 12 required + 9 optional + 2 conditional**

burn-flex has significantly fewer dependencies, with no dependency on ndarray itself, no macro
utility crates, and no BLAS bindings.

---

## 13. Codebase Size

| Metric         | burn-flex     | burn-ndarray |
| -------------- | ------------- | ------------ |
| Source files   | 38            | 37           |
| Total lines    | ~23,500       | ~11,400      |
| ops/ directory | ~19,700 lines | ~8,200 lines |
| SIMD module    | ~1,200 lines  | ~2,100 lines |

burn-flex has roughly 2x the code. This is because:

1. Flex implements all ops from scratch (ndarray delegates to the ndarray crate's built-in ops)
2. Flex has dedicated optimized implementations (pool, conv, reduce, cumulative, gather/scatter)
3. Flex has more comprehensive dtype handling (f16/bf16 paths for every op)
4. Flex has explicit contiguous/non-contiguous fast paths throughout

---

## 14. Testing

| Aspect                | burn-flex                                                  | burn-ndarray            |
| --------------------- | ---------------------------------------------------------- | ----------------------- |
| burn-backend-tests    | **All pass** (6 feature flag combos)                       | All pass                |
| burn-no-std-tests     | **Pass** (MNIST inference)                                 | Not explicitly verified |
| ONNX model checks     | **All pass**                                               | All pass                |
| Real model inference  | **ALBERT, MiniLM**                                         | Not documented          |
| Feature combos tested | no-default, simd, std, std+simd, std+rayon, std+simd+rayon | Default                 |
| Edge-case robustness  | Integer overflow, rounding, zero-size, invalid params      | Standard                |
| Embedded builds       | thumbv6m, thumbv7m, wasm32                                 | wasm32                  |

---

## 15. Performance Summary

All benchmarks on Apple M3 Max, default features enabled.

### Compute Performance

Genuine algorithmic and library improvements:

| Category         | Flex vs NdArray      | Why                                       |
| ---------------- | -------------------- | ----------------------------------------- |
| Binary ops (f32) | **2.4-3.9x faster**  | Arc COW avoids allocation; 3x less memory |
| Binary ops (i64) | **1.5-6.4x faster**  | Same COW benefits                         |
| Matmul (square)  | **1.1-3.4x faster**  | gemm > matrixmultiply                     |
| Matmul (batched) | **1.8-3.2x faster**  | Better batch parallelism                  |
| Attention        | **1.2-2.4x faster**  | Flash attention, 2-8.5x lower peak memory |
| Conv2d           | **1.2-4.0x faster**  | im2col+gemm vs direct                     |
| Conv1d           | **4.3-9.6x faster**  | Unified 3D avoids overhead                |
| Pooling          | **1.2-3.1x faster**  | Unified 3D, better parallelism            |
| Interpolation    | **1.2-3.6x faster**  | Direct computation vs intermediates       |
| Reductions       | **1.6-5.1x faster**  | Zero-alloc SIMD single-pass               |
| Cumulative       | **3.1-97x faster**   | Blocked scan, scalar accumulator          |
| Gather/scatter   | **1.6-9.8x faster**  | Direct indexing                           |
| Unary            | **1.1-2.7x faster**  | In-place mutation when possible           |
| Comparisons      | **2.1-3.9x faster**  | SIMD + compact u8 output                  |
| Int cast         | **5.0-7.6x faster**  | Direct byte reinterpretation              |
| Quantize         | **1.6x faster**      | Fused 2-pass implementation               |
| Concatenation    | **3.6-16.3x faster** | Direct memcpy vs slice_assign             |

### Structural Improvements

These reflect changes in how operations are _represented and executed_, not pure compute speedups.
burn-ndarray eagerly materializes data where burn-flex uses zero-copy views or separated storage.

| Category      | Improvement        | What changed                                                 |
| ------------- | ------------------ | ------------------------------------------------------------ |
| Dequantize    | **135-232x**       | Direct `scale * x_q` vs reparsing `QuantizedBytes` each call |
| Quantized ops | **2.9-117x**       | Dominated by fast dequantize path                            |
| Slice/narrow  | **2.1-2,100x**     | Zero-copy strided view vs potential data copy                |
| Unfold        | **1,200-166,000x** | O(1) strided view vs O(n) full materialization               |
| Expand        | **550-2,600x**     | Zero-copy broadcast (stride=0) vs data copy                  |

> **Note on quantization**: burn-ndarray simulates quantization by dequantizing to f32 for most
> operations. The quantized speedups reflect the difference between simulated and native execution,
> not equivalent algorithms running at different speeds.

### Where NdArray Wins

| Category                   | NdArray advantage | Reason                                      |
| -------------------------- | ----------------- | ------------------------------------------- |
| bool_not/bool_and          | ~20% faster       | ndarray's vectorized mapv is well-optimized |
| int_powf_scalar            | ~10% faster       | ndarray's vectorized internals              |
| Transposed i64 add (large) | ~7% faster        | ndarray handles non-contiguous well         |
| Deform conv (medium)       | ~30% faster       | NdArray has optimized deform conv path      |
| Max pool 5x5               | ~17% faster       | Specific kernel size advantage              |

These are specific edge cases where NdArray's ndarray-based internals have an advantage.

---

## 16. Why Replace burn-ndarray?

The [ndarray](https://crates.io/crates/ndarray) crate has been slow to accept contributions and
evolve. Burn's CPU backend inherits these constraints:

- **No f16/bf16**: Models using half-precision weights must convert to f32. An f16 PR has been open
  for a long time with no clear timeline.
- **6-dimension limit**: Hard-coded in reshape macros, cannot be fixed without upstream changes.
- **Unsigned strides**: `usize`-only strides make zero-copy flip impossible.
- **Simulated quantization**: No native quantized storage; dequantize/requantize on every op.
- **COW limitations**: `NdArrayStorage::Borrowed` always returns false for `is_unique()`, preventing
  in-place mutation of externally loaded data.

burn-flex was built to address these gaps without waiting on upstream. It is not intended to compete
with CubeCL CPU, which targets high-performance computation through operator fusion and just-in-time
compilation via LLVM. The goal is to provide a lightweight, portable replacement for burn-ndarray
that works today on platforms CubeCL CPU cannot target (no_std, WASM, embedded).

## 17. What burn-flex Adds

1. **f16/bf16 support**: Native arithmetic on half-precision types. Enables running models that use
   f16 weights without conversion.

2. **No dimension limit**: Arbitrary tensor rank (ndarray is limited to 6).

3. **Zero-copy flip/unfold/expand**: Signed strides enable O(1) flip. Unfold returns a strided view
   instead of materializing all windows.

4. **Unified 3D conv/pool**: Single implementation covers 1D/2D/3D, reducing code paths and
   potential for inconsistencies.

5. **Native quantization**: Stores scales separately for direct `scale * x_q` dequantization instead
   of reparsing packed bytes on every access. Zero-copy layout ops on quantized tensors.

6. **Fewer dependencies**: 7 required deps vs 12. No ndarray, no matrixmultiply, no paste, no
   const-random, no BLAS bindings.

7. **Simpler type system**: `Flex` vs `NdArray<E, I, Q>`. No generic parameters, no element trait
   hierarchy (`FloatNdArrayElement`, `IntNdArrayElement`, `NdArrayElement`, `ExpElement`).

8. **Real FFT**: Forward (rfft) and inverse (irfft) real FFT with complex packing, SIMD
   butterflies, and compile-time twiddle tables. Works in `no_std` (rustfft/realfft require std).
   NdArray implements neither.

---

## 18. What burn-ndarray Has That burn-flex Does Not

1. **BLAS acceleration**: Feature flags for Accelerate (macOS), OpenBLAS, and Netlib BLAS. These can
   outperform gemm for very large matmuls on specific hardware. burn-flex relies solely on the gemm
   crate.

2. **SIMD conv/pool kernels**: burn-ndarray has dedicated macerator-based SIMD kernels for
   convolution and pooling. burn-flex delegates to gemm's SIMD.

3. **export_tests feature**: burn-ndarray serves as a reference implementation for some burn-cubecl
   kernels via `export_tests`.

---

## 19. Migration Path

For Burn users switching from burn-ndarray to burn-flex:

| Change         | Details                                            |
| -------------- | -------------------------------------------------- |
| Type parameter | `NdArray<f32>` becomes `Flex`                      |
| Device         | `NdArrayDevice::Cpu` becomes `FlexDevice`          |
| Feature flags  | `multi-threads` becomes `rayon`                    |
| BLAS features  | No equivalent (gemm handles matmul)                |
| Autodiff       | Use `burn_autodiff::Autodiff<Flex>` (same pattern) |
| f16/bf16       | Works out of the box (new capability)              |
| Quantization   | Same API, faster execution                         |
| Tests          | Same burn-backend-tests suite passes               |

---

## 20. Conclusion

burn-flex is a from-scratch replacement for burn-ndarray, motivated by ndarray's lack of f16/bf16
support, 6-dimension limit, simulated quantization, and slow pace of upstream development. It
implements all required Backend traits (FloatTensorOps, IntTensorOps, BoolTensorOps, QTensorOps,
ModuleOps, ActivationOps, TransactionOps) and passes the same test suite.

Performance gains come in two forms: compute improvements (1.1-9.7x) from better libraries and
algorithms, and structural improvements (up to 166,000x) from representing operations as zero-copy
views instead of eagerly materializing data. Memory usage is significantly reduced through Arc-based
COW and in-place mutation.

The only capabilities lost are optional BLAS acceleration (replaced by the gemm crate, which is
faster in most benchmarks) and the `export_tests` reference implementation feature.
