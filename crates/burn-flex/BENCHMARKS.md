# Flex Benchmarks

Flex CPU backend timings on an Apple M3 Max, with the default features (`std`, `simd`, `rayon`).
Each table compares the current run with the April 2026 run, the Flex column of the former
Flex-vs-NdArray comparison. `burn-ndarray` was removed after the 0.22 release; that comparison
remains in [`release/0.22`](https://github.com/tracel-ai/burn/blob/release/0.22/crates/burn-flex/BENCHMARKS.md).

**Date**: 2026-11-05, commit `5561bf539` (April 2026 baseline: 2026-04-06)

## How to Read

- **Now** and **Apr 2026** are divan medians (lower is better).
- **Change** = Apr 2026 median / Now median. Bold means faster now; `new` marks benches added
  since April.
- **Mem** = peak allocation of the current run (`max alloc` from divan).
- The April run called `Tensor<Flex, D>` directly. The benches now live in `burn-backend-tests` and
  go through `burn-tensor`'s device dispatch, which adds roughly 100 ns per op. Ops that take
  under a microsecond (slicing, `unfold`, casts that only relabel, `expand`) show that overhead as
  a slowdown; it is not a change in the kernels.
- **Now** is the faster median of two full passes on AC power. Other work on the machine can slow
  a pass, so a single pass can read high: the passes differ by 2% in the median row and by up to
  2.4x in the worst (attention decode and adaptive average pooling). Treat differences within about
  10% as noise.

---

## Binary Operations (f32)

| Operation   | Size | Now     | Apr 2026 | Change   | Mem     |
| ----------- | ---- | ------- | -------- | -------- | ------- |
| add         | 4K   | 537 ns  | 389 ns   | 0.7x     | 16.5 KB |
| add         | 64K  | 7.44 us | 7.36 us  | ~1x      | 262 KB  |
| add         | 1M   | 112 us  | 83.9 us  | 0.8x     | 4.19 MB |
| mul         | 4K   | 529 ns  | 382 ns   | 0.7x     | 16.5 KB |
| mul         | 64K  | 7.45 us | 7.40 us  | ~1x      | 262 KB  |
| mul         | 1M   | 113 us  | 115 us   | ~1x      | 4.19 MB |
| div         | 1M   | 113 us  | 115 us   | ~1x      | 4.19 MB |
| add_scalar  | 1M   | 74.4 us | 78.7 us  | ~1x      | 4.19 MB |
| mul_scalar  | 1M   | 74.4 us | 75.8 us  | ~1x      | 4.19 MB |
| powf        | 64K  | 192 us  | 197 us   | ~1x      | 262 KB  |
| powf        | 1M   | 3.13 ms | 3.17 ms  | ~1x      | 4.19 MB |
| powf_scalar | 1M   | 2.96 ms | 3.23 ms  | ~1x      | 4.19 MB |
| atan2       | 64K  | 66.3 us | 143 us   | **2.2x** | 262 KB  |
| atan2       | 1M   | 1.07 ms | 2.33 ms  | **2.2x** | 4.19 MB |
| fmod        | 64K  | 109 us  | -        | new      | 262 KB  |
| fmod        | 1M   | 1.76 ms | -        | new      | 4.19 MB |
| fmod_scalar | 1M   | 2.91 ms | -        | new      | 4.19 MB |

### Transposed

| Operation | Size      | Now     | Apr 2026 | Change | Mem     |
| --------- | --------- | ------- | -------- | ------ | ------- |
| add       | 256x256   | 55.4 us | 48.5 us  | 0.9x   | 262 KB  |
| add       | 1024x1024 | 956 us  | 1.00 ms  | ~1x    | 4.19 MB |

---

## Binary Operations (int)

| Operation      | Size | Now     | Apr 2026 | Change | Mem     |
| -------------- | ---- | ------- | -------- | ------ | ------- |
| int_add        | 4K   | 511 ns  | 361 ns   | 0.7x   | 16.4 KB |
| int_add        | 64K  | 7.40 us | 7.40 us  | ~1x    | 262 KB  |
| int_add        | 1M   | 113 us  | 115 us   | ~1x    | 4.19 MB |
| int_mul        | 4K   | 506 ns  | 366 ns   | 0.7x   | 16.4 KB |
| int_mul        | 64K  | 7.44 us | 7.40 us  | ~1x    | 262 KB  |
| int_mul        | 1M   | 112 us  | 115 us   | ~1x    | 4.19 MB |
| int_div        | 1M   | 585 us  | 604 us   | ~1x    | 4.19 MB |
| int_add_scalar | 1M   | 74.5 us | 75.8 us  | ~1x    | 4.19 MB |
| int_mul_scalar | 1M   | 74.5 us | 75.7 us  | ~1x    | 4.19 MB |

### Int Power

| Operation | Size     | Now     | Apr 2026 | Change | Mem     |
| --------- | -------- | ------- | -------- | ------ | ------- |
| int_powi  | 256x256  | 91.4 us | 95.6 us  | ~1x    | 262 KB  |
| int_powi  | 1024x256 | 356 us  | 336 us   | ~1x    | 1.05 MB |

### Transposed (int)

| Operation | Size      | Now     | Apr 2026 | Change | Mem     |
| --------- | --------- | ------- | -------- | ------ | ------- |
| int_add   | 256x256   | 55.4 us | 55.6 us  | ~1x    | 262 KB  |
| int_add   | 1024x1024 | 947 us  | 996 us   | ~1x    | 4.19 MB |

---

## Int Cast

| Operation  | Size      | Now     | Apr 2026 | Change | Mem     |
| ---------- | --------- | ------- | -------- | ------ | ------- |
| i64 to i8  | 256x256   | 3.01 us | 3.19 us  | ~1x    | 65.6 KB |
| i64 to i32 | 64x64     | 92.6 ns | 16.8 ns  | 0.2x   | 16.0 B  |
| i64 to i32 | 256x256   | 98.5 ns | 13.6 ns  | 0.1x   | 16.0 B  |
| i64 to i32 | 1024x1024 | 93.2 ns | 13.6 ns  | 0.1x   | 16.0 B  |

---

## Int Random

| Operation | Size       | Now     | Apr 2026 | Change   | Mem     |
| --------- | ---------- | ------- | -------- | -------- | ------- |
| uniform   | 64x64      | 18.3 us | 20.9 us  | **1.1x** | 16.5 KB |
| uniform   | 256x256    | 310 us  | 334 us   | ~1x      | 262 KB  |
| uniform   | 1024x1024  | 5.01 ms | 5.36 ms  | ~1x      | 4.19 MB |
| uniform   | 16x128x128 | 1.25 ms | 1.35 ms  | ~1x      | 1.05 MB |

---

## Matrix Multiplication

### Square (f32)

| Size      | Now     | Apr 2026 | Change   | Mem     |
| --------- | ------- | -------- | -------- | ------- |
| 64x64     | 5.68 us | 6.06 us  | ~1x      | 33.6 KB |
| 128x128   | 42.6 us | 43.8 us  | ~1x      | 328 KB  |
| 256x256   | 149 us  | 166 us   | **1.1x** | 524 KB  |
| 512x512   | 564 us  | 579 us   | ~1x      | 2.10 MB |
| 1024x1024 | 2.64 ms | 2.69 ms  | ~1x      | 8.39 MB |

### Rectangular (f32)

| Shape               | Now    | Apr 2026 | Change   | Mem     |
| ------------------- | ------ | -------- | -------- | ------- |
| 512x64 x 64x512     | 152 us | 167 us   | ~1x      | 1.18 MB |
| 256x512 x 512x256   | 228 us | 265 us   | **1.2x** | 786 KB  |
| 128x1024 x 1024x128 | 159 us | 190 us   | **1.2x** | 590 KB  |

### Transposed (256x256)

| Config          | Now    | Apr 2026 | Change | Mem    |
| --------------- | ------ | -------- | ------ | ------ |
| LHS transposed  | 154 us | 140 us   | ~1x    | 524 KB |
| RHS transposed  | 158 us | 158 us   | ~1x    | 788 KB |
| Both transposed | 157 us | 165 us   | ~1x    | 786 KB |

### Batched (f32)

| Shape              | Now     | Apr 2026 | Change   | Mem     |
| ------------------ | ------- | -------- | -------- | ------- |
| 8x 64x64           | 54.7 us | 57.6 us  | ~1x      | 131 KB  |
| 32x 64x64          | 71.0 us | 67.0 us  | ~1x      | 524 KB  |
| 16x 128x128        | 204 us  | 267 us   | **1.3x** | 1.05 MB |
| 12x 512x64 (heads) | 573 us  | 777 us   | **1.4x** | 12.6 MB |

### Broadcast (f32)

| Shape                     | Now     | Apr 2026 | Change | Mem     |
| ------------------------- | ------- | -------- | ------ | ------- |
| [1,64,64] x [8,64,64]     | 51.2 us | 47.9 us  | ~1x    | 131 KB  |
| [8,64,64] x [1,64,64]     | 51.5 us | 54.0 us  | ~1x    | 131 KB  |
| [2,1,32,32] x [1,4,32,32] | 7.11 us | 7.00 us  | ~1x    | 34.4 KB |
| [4,1,64,64] x [1,4,64,64] | 54.2 us | 50.0 us  | ~1x    | 262 KB  |

### Transformer Shapes (f16, weight grad in f32)

| Shape                           | Now     | Apr 2026 | Change | Mem     |
| ------------------------------- | ------- | -------- | ------ | ------- |
| decode 1x4096 x 4096x4096       | 602 us  | -        | new    | 8.30 KB |
| decode 8x4096 x 4096x12288      | 4.04 ms | -        | new    | 264 KB  |
| prefill 512x4096 x 4096x4096    | 9.29 ms | -        | new    | 8.39 MB |
| prefill 512x4096 x 4096x12288   | 26.0 ms | -        | new    | 16.8 MB |
| weight grad 4096x512 x 512x4096 | 18.3 ms | -        | new    | 75.5 MB |

### Integer (i32)

| Shape       | Now     | Apr 2026 | Change   | Mem     |
| ----------- | ------- | -------- | -------- | ------- |
| 64x64       | 25.4 us | 30.2 us  | **1.2x** | 32.9 KB |
| 128x128     | 171 us  | 196 us   | **1.1x** | 131 KB  |
| 256x256     | 467 us  | 1.90 ms  | **4.1x** | 524 KB  |
| 512x512     | 2.24 ms | 18.3 ms  | **8.2x** | 2.10 MB |
| 8x 64x64    | 128 us  | -        | new      | 262 KB  |
| 16x 128x128 | 918 us  | -        | new      | 2.10 MB |
| 1Mx8 x 8x1  | 311 us  | -        | new      | 4.19 MB |

---

## Slice Operations

### Basic Slicing

| Operation | Size      | Now    | Apr 2026 | Change | Mem    |
| --------- | --------- | ------ | -------- | ------ | ------ |
| slice 1D  | 1K        | 173 ns | 112 ns   | 0.6x   | 9.5 B  |
| slice 1D  | 1M        | 170 ns | 105 ns   | 0.6x   | 9.5 B  |
| slice 2D  | 256x256   | 186 ns | 120 ns   | 0.6x   | 22.0 B |
| slice 2D  | 1024x1024 | 192 ns | 115 ns   | 0.6x   | 19.0 B |
| slice 3D  | 64x64x64  | 199 ns | 148 ns   | 0.7x   | 33.0 B |

### Narrow

| Operation   | Size      | Now    | Apr 2026 | Change | Mem    |
| ----------- | --------- | ------ | -------- | ------ | ------ |
| narrow dim0 | 256x256   | 231 ns | 141 ns   | 0.6x   | 22.0 B |
| narrow dim0 | 1024x1024 | 227 ns | 142 ns   | 0.6x   | 19.0 B |
| narrow dim1 | 256x256   | 221 ns | 128 ns   | 0.6x   | 19.0 B |

### Slice Assignment

| Operation | Size      | Now     | Apr 2026 | Change | Mem     |
| --------- | --------- | ------- | -------- | ------ | ------- |
| assign 1D | 1K        | 408 ns  | 304 ns   | 0.7x   | 4.18 KB |
| assign 2D | 256x256   | 5.15 us | 5.39 us  | ~1x    | 262 KB  |
| assign 2D | 1024x1024 | 71.9 us | 74.9 us  | ~1x    | 4.19 MB |

### Slice Fill

| Region   | Of         | Now     | Apr 2026 | Change | Mem     |
| -------- | ---------- | ------- | -------- | ------ | ------- |
| 512      | 1K         | 581 ns  | -        | new    | 4.19 KB |
| 128x128  | 256x256    | 5.59 us | -        | new    | 262 KB  |
| 244x224  | 488x448    | 15.1 us | -        | new    | 875 KB  |
| 512x512  | 1024x1024  | 66.2 us | -        | new    | 4.19 MB |
| 64x64x64 | 64x128x128 | 74.3 us | -        | new    | 4.19 MB |

### Transposed Slicing

| Size      | Now    | Apr 2026 | Change | Mem    |
| --------- | ------ | -------- | ------ | ------ |
| 256x256   | 185 ns | 98.2 ns  | 0.5x   | 19.0 B |
| 1024x1024 | 185 ns | 98.2 ns  | 0.5x   | 19.0 B |

### Slice with Step

| Operation | Size      | Now    | Apr 2026 | Change | Mem    |
| --------- | --------- | ------ | -------- | ------ | ------ |
| step2 1D  | 1K        | 165 ns | 87.1 ns  | 0.5x   | 9.5 B  |
| step2 1D  | 1M        | 161 ns | 76.7 ns  | 0.5x   | 9.5 B  |
| step2 2D  | 1024x1024 | 186 ns | 103 ns   | 0.6x   | 19.0 B |
| step4 2D  | 256x256   | 184 ns | 101 ns   | 0.6x   | 19.0 B |

---

## Concatenation

### Cat (dim 0, contiguous memcpy fast path)

| Tensors | Size     | Now     | Apr 2026 | Change   | Mem     |
| ------- | -------- | ------- | -------- | -------- | ------- |
| 4x      | 256x256  | 13.2 us | 16.0 us  | **1.2x** | 1.05 MB |
| 4x      | 1024x256 | 50.2 us | 57.9 us  | **1.2x** | 4.20 MB |
| 16x     | 64x64    | 4.00 us | 4.11 us  | ~1x      | 265 KB  |
| 4x      | 16K (1D) | 3.09 us | 3.47 us  | **1.1x** | 262 KB  |

### Cat (dim 1, general path)

| Tensors | Size    | Now     | Apr 2026 | Change   | Mem     |
| ------- | ------- | ------- | -------- | -------- | ------- |
| 4x      | 256x64  | 6.16 us | 6.92 us  | **1.1x** | 263 KB  |
| 4x      | 1024x64 | 22.2 us | 25.5 us  | **1.1x** | 1.05 MB |

---

## Reduce Operations

### Full Tensor Sum

| Size | Now     | Apr 2026 | Change   | Mem    |
| ---- | ------- | -------- | -------- | ------ |
| 1K   | 190 ns  | 118 ns   | 0.6x     | 76.2 B |
| 64K  | 2.74 us | 3.23 us  | **1.2x** | 80.0 B |
| 1M   | 39.6 us | 43.3 us  | ~1x      | 84.0 B |

### Full Tensor Max

| Size | Now     | Apr 2026 | Change | Mem    |
| ---- | ------- | -------- | ------ | ------ |
| 1K   | 276 ns  | 207 ns   | 0.7x   | 76.5 B |
| 64K  | 8.28 us | 8.78 us  | ~1x    | 84.0 B |
| 1M   | 139 us  | 139 us   | ~1x    | 84.0 B |

### Full Tensor Min

| Size | Now     | Apr 2026 | Change | Mem    |
| ---- | ------- | -------- | ------ | ------ |
| 1K   | 279 ns  | 278 ns   | ~1x    | 76.5 B |
| 64K  | 8.86 us | 9.15 us  | ~1x    | 84.0 B |
| 1M   | 139 us  | 142 us   | ~1x    | 84.0 B |

### Int Max

| Size      | Now     | Apr 2026 | Change | Mem    |
| --------- | ------- | -------- | ------ | ------ |
| 256x256   | 2.84 us | 2.84 us  | ~1x    | 84.0 B |
| 1024x1024 | 41.7 us | 42.2 us  | ~1x    | 92.0 B |

### Sum Along Dimension

| Shape     | Dim | Now     | Apr 2026 | Change | Mem     |
| --------- | --- | ------- | -------- | ------ | ------- |
| 256x256   | 0   | 5.19 us | 5.06 us  | ~1x    | 2.10 KB |
| 256x256   | 1   | 2.82 us | 2.77 us  | ~1x    | 1.12 KB |
| 1024x1024 | 0   | 78.6 us | 80.0 us  | ~1x    | 8.24 KB |
| 1024x1024 | 1   | 39.9 us | 42.2 us  | ~1x    | 4.21 KB |
| 1Mx16     | 1   | 1.98 ms | -        | new    | 4.19 MB |

### 3D Sum (Batched)

| Shape      | Dim | Now     | Apr 2026 | Change | Mem     |
| ---------- | --- | ------- | -------- | ------ | ------- |
| 32x256x256 | 1   | 152 us  | 156 us   | ~1x    | 65.6 KB |
| 32x256x256 | 2   | 81.0 us | 86.2 us  | ~1x    | 32.9 KB |

### Sum Transposed

| Size      | Now     | Apr 2026 | Change   | Mem    |
| --------- | ------- | -------- | -------- | ------ |
| 256x256   | 2.80 us | 3.27 us  | **1.2x** | 94.0 B |
| 1024x1024 | 39.7 us | 41.4 us  | ~1x      | 112 B  |

### Sum Dim on Transposed

| Size      | Dim | Now     | Apr 2026 | Change | Mem     |
| --------- | --- | ------- | -------- | ------ | ------- |
| 256x256   | 0   | 2.78 us | 2.79 us  | ~1x    | 1.12 KB |
| 1024x1024 | 0   | 39.9 us | 41.5 us  | ~1x    | 4.21 KB |

### Mean Along Dimension

| Shape     | Dim | Now     | Apr 2026 | Change   | Mem     |
| --------- | --- | ------- | -------- | -------- | ------- |
| 256x256   | 1   | 2.67 us | 2.90 us  | ~1x      | 1.12 KB |
| 1024x1024 | 1   | 38.0 us | 42.5 us  | **1.1x** | 4.21 KB |

### f16 Mean

| Shape      | Dim | Now     | Apr 2026 | Change | Mem     |
| ---------- | --- | ------- | -------- | ------ | ------- |
| 64K        | -   | 20.2 us | -        | new    | 262 KB  |
| 1M         | -   | 319 us  | -        | new    | 4.19 MB |
| 256x256    | 0   | 22.8 us | -        | new    | 264 KB  |
| 256x256    | 1   | 20.5 us | -        | new    | 264 KB  |
| 1024x1024  | 0   | 358 us  | -        | new    | 4.20 MB |
| 1024x1024  | 1   | 319 us  | -        | new    | 4.20 MB |
| 32x256x256 | 1   | 712 us  | -        | new    | 8.45 MB |

### Argmax

| Shape     | Dim | Now     | Apr 2026 | Change | Mem     |
| --------- | --- | ------- | -------- | ------ | ------- |
| 1K        | -   | 724 ns  | 752 ns   | ~1x    | 98.0 B  |
| 256x256   | 1   | 95.5 us | 66.7 us  | 0.7x   | 3.25 KB |
| 1024x1024 | 1   | 117 us  | 120 us   | ~1x    | 12.5 KB |

---

## Cumulative Operations

### Cumsum

| Shape     | Dim | Now     | Apr 2026 | Change | Mem     |
| --------- | --- | ------- | -------- | ------ | ------- |
| 1K        | 0   | 844 ns  | 838 ns   | ~1x    | 4.17 KB |
| 64K       | 0   | 41.5 us | 45.2 us  | ~1x    | 262 KB  |
| 1M        | 0   | 659 us  | 719 us   | ~1x    | 4.19 MB |
| 256x256   | 0   | 11.2 us | 11.3 us  | ~1x    | 263 KB  |
| 256x256   | 1   | 41.9 us | 42.6 us  | ~1x    | 262 KB  |
| 1024x1024 | 1   | 648 us  | 709 us   | ~1x    | 4.19 MB |

### Cumprod

| Shape   | Dim | Now     | Apr 2026 | Change | Mem     |
| ------- | --- | ------- | -------- | ------ | ------- |
| 1K      | 0   | 1.24 us | 1.27 us  | ~1x    | 4.17 KB |
| 256x256 | 1   | 60.7 us | 66.3 us  | ~1x    | 262 KB  |

### Cummin

| Shape     | Dim | Now     | Apr 2026 | Change | Mem     |
| --------- | --- | ------- | -------- | ------ | ------- |
| 1K        | 0   | 1.72 us | 1.79 us  | ~1x    | 4.17 KB |
| 256x256   | 1   | 93.8 us | 102 us   | ~1x    | 262 KB  |
| 1024x1024 | 1   | 1.56 ms | 1.71 ms  | ~1x    | 4.19 MB |

### Cummax

| Shape     | Dim | Now     | Apr 2026 | Change | Mem     |
| --------- | --- | ------- | -------- | ------ | ------- |
| 1K        | 0   | 1.74 us | 1.82 us  | ~1x    | 4.18 KB |
| 256x256   | 1   | 93.9 us | 102 us   | ~1x    | 262 KB  |
| 1024x1024 | 1   | 1.56 ms | 1.70 ms  | ~1x    | 4.19 MB |

### 3D Cumsum (Batched)

| Shape    | Dim | Now     | Apr 2026 | Change | Mem    |
| -------- | --- | ------- | -------- | ------ | ------ |
| 32x64x64 | 1   | 23.5 us | 24.1 us  | ~1x    | 524 KB |
| 32x64x64 | 2   | 66.7 us | 68.0 us  | ~1x    | 524 KB |

---

## Gather/Scatter Operations

### Gather

| Shape     | Dim | Now     | Apr 2026 | Change   | Mem     |
| --------- | --- | ------- | -------- | -------- | ------- |
| 256x256   | 0   | 30.1 us | 32.9 us  | ~1x      | 393 KB  |
| 256x256   | 1   | 30.8 us | 33.8 us  | ~1x      | 393 KB  |
| 1024x1024 | 1   | 235 us  | 273 us   | **1.2x** | 6.29 MB |

### Scatter Add

| Shape     | Dim | Now     | Apr 2026 | Change | Mem     |
| --------- | --- | ------- | -------- | ------ | ------- |
| 256x256   | 1   | 33.0 us | 35.7 us  | ~1x    | 524 KB  |
| 1024x1024 | 1   | 543 us  | 563 us   | ~1x    | 8.39 MB |

### Select

| Shape     | Dim | Now     | Apr 2026 | Change | Mem     |
| --------- | --- | ------- | -------- | ------ | ------- |
| 256x256   | 0   | 2.12 us | 2.02 us  | ~1x    | 132 KB  |
| 256x256   | 1   | 26.0 us | 26.3 us  | ~1x    | 132 KB  |
| 1024x1024 | 0   | 26.1 us | 26.8 us  | ~1x    | 2.10 MB |

### Bool Select

| Shape    | Indices | Now     | Apr 2026 | Change   | Mem     |
| -------- | ------- | ------- | -------- | -------- | ------- |
| 256x256  | 128     | 773 ns  | 935 ns   | **1.2x** | 33.1 KB |
| 1024x256 | 512     | 2.37 us | 2.89 us  | **1.2x** | 133 KB  |

### Select Add

| Shape     | Dim | Now     | Apr 2026 | Change | Mem     |
| --------- | --- | ------- | -------- | ------ | ------- |
| 256x256   | 0   | 6.92 us | 7.35 us  | ~1x    | 263 KB  |
| 1024x1024 | 0   | 99.0 us | 103 us   | ~1x    | 4.20 MB |

---

## Unary Operations

F32 `exp`, `log`, `sin`, `cos` and `tanh` on contiguous tensors now run with SIMD kernels; the
transposed `exp` rows take the strided scalar path.

### Basic Math

| Operation | Size | Now     | Apr 2026 | Change   | Mem     |
| --------- | ---- | ------- | -------- | -------- | ------- |
| exp       | 4K   | 2.30 us | 5.07 us  | **2.2x** | 16.5 KB |
| exp       | 64K  | 33.8 us | 80.5 us  | **2.4x** | 262 KB  |
| exp       | 1M   | 543 us  | 1.31 ms  | **2.4x** | 4.19 MB |
| log       | 4K   | 3.53 us | 6.74 us  | **1.9x** | 16.5 KB |
| log       | 64K  | 53.8 us | 106 us   | **2.0x** | 262 KB  |
| log       | 1M   | 860 us  | 1.72 ms  | **2.0x** | 4.19 MB |
| sqrt      | 4K   | 677 ns  | 612 ns   | ~1x      | 16.4 KB |
| sqrt      | 64K  | 8.36 us | 9.03 us  | ~1x      | 262 KB  |
| sqrt      | 1M   | 130 us  | 142 us   | ~1x      | 4.19 MB |
| abs       | 1M   | 69.4 us | 75.8 us  | ~1x      | 4.19 MB |
| recip     | 1M   | 69.3 us | 75.5 us  | ~1x      | 4.19 MB |

### Trigonometric

| Operation | Size | Now     | Apr 2026 | Change   | Mem     |
| --------- | ---- | ------- | -------- | -------- | ------- |
| sin       | 4K   | 3.24 us | 5.65 us  | **1.7x** | 16.5 KB |
| sin       | 64K  | 48.7 us | 89.2 us  | **1.8x** | 262 KB  |
| sin       | 1M   | 779 us  | 1.45 ms  | **1.9x** | 4.19 MB |
| cos       | 4K   | 3.11 us | 6.57 us  | **2.1x** | 16.5 KB |
| cos       | 1M   | 790 us  | 1.68 ms  | **2.1x** | 4.19 MB |
| tanh      | 4K   | 4.11 us | 7.07 us  | **1.7x** | 16.5 KB |
| tanh      | 64K  | 62.8 us | 112 us   | **1.8x** | 262 KB  |
| tanh      | 1M   | 1.00 ms | 1.80 ms  | **1.8x** | 4.19 MB |

### Transposed (Non-contiguous)

| Operation | Size      | Now     | Apr 2026 | Change | Mem     |
| --------- | --------- | ------- | -------- | ------ | ------- |
| exp       | 256x256   | 78.4 us | 80.1 us  | ~1x    | 262 KB  |
| exp       | 1024x1024 | 1.27 ms | 1.31 ms  | ~1x    | 4.19 MB |

---

## Softmax (f32)

`trait_hook` calls the backend's softmax; `decomposed` builds it from exp, sum and div.

| Axis       | Shape                   | Path       | Now     | Apr 2026 | Change | Mem     |
| ---------- | ----------------------- | ---------- | ------- | -------- | ------ | ------- |
| last       | BERT, seq 512, 12 heads | trait_hook | 385 us  | -        | new    | 12.6 MB |
| last       | BERT, seq 512, 12 heads | decomposed | 2.38 ms | -        | new    | 12.6 MB |
| last       | d=2048                  | trait_hook | 259 us  | -        | new    | 4.19 MB |
| last       | d=2048                  | decomposed | 903 us  | -        | new    | 4.20 MB |
| dim 1 of 3 | -                       | trait_hook | 766 us  | -        | new    | 16.8 MB |
| dim 1 of 3 | -                       | decomposed | 2.34 ms | -        | new    | 8.40 MB |

---

## Layer Norm (f32)

`fused_trait` calls the backend's layer norm; `decomposed` builds it from mean, sub, powi, div,
mul and add. The composite rows include the NCHW to NHWC reshape ConvNeXt does around it.

| Shape                 | Path        | Now     | Apr 2026 | Change | Mem     |
| --------------------- | ----------- | ------- | -------- | ------ | ------- |
| ConvNeXt c48, 244x224 | fused_trait | 246 us  | -        | new    | 10.5 MB |
| ConvNeXt c48, 244x224 | decomposed  | 1.39 ms | -        | new    | 21.2 MB |
| ConvNeXt c96, 122x112 | fused_trait | 180 us  | -        | new    | 5.25 MB |
| ConvNeXt c96, 122x112 | decomposed  | 629 us  | -        | new    | 10.5 MB |
| ConvNeXt c192, 61x56  | fused_trait | 94.0 us | -        | new    | 2.62 MB |
| ConvNeXt c192, 61x56  | decomposed  | 306 us  | -        | new    | 5.26 MB |
| ConvNeXt c384, 30x28  | fused_trait | 66.3 us | -        | new    | 1.29 MB |
| ConvNeXt c384, 30x28  | decomposed  | 155 us  | -        | new    | 2.58 MB |
| d768, seq 512, b2     | fused_trait | 108 us  | -        | new    | 3.15 MB |
| d768, seq 512, b2     | decomposed  | 355 us  | -        | new    | 6.29 MB |
| c48, 244x224          | composite   | 3.79 ms | -        | new    | 21.2 MB |
| c96, 122x112          | composite   | 1.79 ms | -        | new    | 10.5 MB |
| c192, 61x56           | composite   | 925 us  | -        | new    | 5.26 MB |
| c384, 30x28           | composite   | 443 us  | -        | new    | 2.58 MB |

### Decomposed Steps

| Step                      | Now     | Apr 2026 | Change | Mem     |
| ------------------------- | ------- | -------- | ------ | ------- |
| mean_dim (last)           | 139 us  | -        | new    | 219 KB  |
| broadcast sub             | 200 us  | -        | new    | 10.5 MB |
| powi_scalar(2)            | 188 us  | -        | new    | 10.5 MB |
| broadcast div             | 206 us  | -        | new    | 10.5 MB |
| broadcast mul (1D)        | 215 us  | -        | new    | 10.5 MB |
| broadcast add (1D)        | 216 us  | -        | new    | 10.5 MB |
| permute only              | 158 ns  | -        | new    | 34.0 B  |
| mean_dim on permuted      | 484 us  | -        | new    | 10.7 MB |
| broadcast sub on permuted | 1.83 ms | -        | new    | 10.5 MB |

---

## Comparison & Boolean Operations

### Tensor-Tensor Comparisons

| Operation | Size | Now     | Apr 2026 | Change | Mem     |
| --------- | ---- | ------- | -------- | ------ | ------- |
| greater   | 4K   | 534 ns  | 431 ns   | 0.8x   | 4.17 KB |
| greater   | 64K  | 6.15 us | 6.48 us  | ~1x    | 65.6 KB |
| greater   | 1M   | 86.7 us | 93 us    | ~1x    | 1.05 MB |
| equal     | 4K   | 565 ns  | 433 ns   | 0.8x   | 4.17 KB |
| equal     | 1M   | 86.6 us | 86 us    | ~1x    | 1.05 MB |
| lower     | 1M   | 86.6 us | 92 us    | ~1x    | 1.05 MB |

### Scalar Comparisons

| Operation    | Size | Now     | Apr 2026 | Change | Mem     |
| ------------ | ---- | ------- | -------- | ------ | ------- |
| greater_elem | 1M   | 51.4 us | 56 us    | ~1x    | 1.05 MB |

### Transposed Comparisons

| Operation | Size      | Now     | Apr 2026 | Change | Mem     |
| --------- | --------- | ------- | -------- | ------ | ------- |
| greater   | 256x256   | 54.0 us | 53.6 us  | ~1x    | 65.6 KB |
| greater   | 1024x1024 | 942 us  | 985 us   | ~1x    | 1.05 MB |

### Broadcast Comparisons

| Operation | Shape     | Now     | Apr 2026 | Change   | Mem     |
| --------- | --------- | ------- | -------- | -------- | ------- |
| greater   | 256x256   | 3.89 us | 7.98 us  | **2.1x** | 65.6 KB |
| greater   | 1024x1024 | 41.5 us | 120 us   | **2.9x** | 1.05 MB |

### Expand (Broadcasting)

| Operation           | Now    | Apr 2026 | Change | Mem    |
| ------------------- | ------ | -------- | ------ | ------ |
| 1x1 to 1000x1000    | 210 ns | 126 ns   | 0.6x   | 17.5 B |
| 1024x1 to 1024x1024 | 195 ns | 110 ns   | 0.6x   | 17.5 B |
| 1x1024 to 1024x1024 | 196 ns | 126 ns   | 0.6x   | 17.5 B |

### Boolean Operations

| Operation | Size | Now     | Apr 2026 | Change | Mem     |
| --------- | ---- | ------- | -------- | ------ | ------- |
| bool_not  | 1M   | 23.8 us | 24.1 us  | ~1x    | 1.05 MB |
| bool_and  | 1M   | 31.8 us | 34.6 us  | ~1x    | 1.05 MB |

---

## Convolutions

### Kernel Size Comparison (4x64x56x56, 64 to 128 channels)

| Kernel | Now     | Apr 2026 | Change   | Mem     |
| ------ | ------- | -------- | -------- | ------- |
| 1x1    | 405 us  | 577 us   | **1.4x** | 6.42 MB |
| 3x3    | 3.74 ms | 3.65 ms  | ~1x      | 13.9 MB |
| 5x5    | 10.3 ms | 8.05 ms  | 0.8x     | 24.0 MB |
| 7x7    | 23.1 ms | 15.7 ms  | 0.7x     | 24.0 MB |

### ResNet Layers (batch=1, 3x3)

| Layer  | Input       | Channels       | Now     | Apr 2026 | Change   | Mem     |
| ------ | ----------- | -------------- | ------- | -------- | -------- | ------- |
| conv1  | 1x3x224x224 | 3 to 64 (k7s2) | 790 us  | 954 us   | **1.2x** | 10.6 MB |
| layer1 | 1x64x56x56  | 64 to 64       | 807 us  | 986 us   | **1.2x** | 8.18 MB |
| layer2 | 1x128x28x28 | 128 to 128     | 713 us  | 1.08 ms  | **1.5x** | 4.60 MB |
| layer3 | 1x256x14x14 | 256 to 256     | 973 us  | 1.65 ms  | **1.7x** | 3.19 MB |
| layer4 | 1x512x7x7   | 512 to 512     | 1.04 ms | 2.71 ms  | **2.6x** | 4.54 MB |

### Small (batch=1, 3x3)

| Input      | Channels | Now     | Apr 2026 | Change   | Mem     |
| ---------- | -------- | ------- | -------- | -------- | ------- |
| 1x3x32x32  | 3 to 16  | 47.7 us | 71.3 us  | **1.5x** | 66.0 KB |
| 1x16x32x32 | 16 to 32 | 146 us  | 219 us   | **1.5x** | 740 KB  |
| 1x32x16x16 | 32 to 64 | 175 us  | 164 us   | ~1x      | 434 KB  |

### Small Channel Counts

| Case          | Input       | Kernel        | Now     | Apr 2026 | Change | Mem     |
| ------------- | ----------- | ------------- | ------- | -------- | ------ | ------- |
| first layer   | 4x3x224x224 | 7x7 s2        | 3.08 ms | -        | new    | 20.3 MB |
| preprocessing | 1x3x488x448 | 5x5, to 8     | 1.30 ms | -        | new    | 7.00 MB |
| downsample    | 1x4x128x128 | 4x4 s2, to 12 | 199 us  | -        | new    | 197 KB  |
| mask          | 1x1x256x256 | 3x3, to 8     | 159 us  | -        | new    | 2.10 MB |
| sobel         | 1x3x488x448 | 3x3           | 457 us  | -        | new    | 2.62 MB |
| sobel         | 1x3x488x448 | 1x3           | 222 us  | -        | new    | 2.62 MB |
| sobel         | 1x3x488x448 | 3x1           | 205 us  | -        | new    | 2.62 MB |
| sobel         | 1x3x488x448 | 1x5           | 314 us  | -        | new    | 2.62 MB |
| sobel         | 1x3x488x448 | 5x1           | 286 us  | -        | new    | 2.62 MB |

### Large Batched (batch=16, 3x3)

| Input         | Channels   | Now     | Apr 2026 | Change   | Mem     |
| ------------- | ---------- | ------- | -------- | -------- | ------- |
| 16x64x128x128 | 64 to 128  | 65.9 ms | 79.8 ms  | **1.2x** | 151 MB  |
| 16x128x64x64  | 128 to 256 | 57.8 ms | 59.8 ms  | ~1x      | 85.1 MB |

### Medium Batched (batch=8, 3x3)

| Input      | Channels  | Now     | Apr 2026 | Change | Mem     |
| ---------- | --------- | ------- | -------- | ------ | ------- |
| 8x3x64x64  | 3 to 64   | 897 us  | 925 us   | ~1x    | 4.64 MB |
| 8x32x64x64 | 32 to 64  | 5.38 ms | 4.67 ms  | 0.9x   | 13.2 MB |
| 8x64x32x32 | 64 to 128 | 4.33 ms | 3.07 ms  | 0.7x   | 6.85 MB |

### Depthwise Conv2d

| Input       | Kernel | Now    | Apr 2026 | Change | Mem     |
| ----------- | ------ | ------ | -------- | ------ | ------- |
| 4x32x56x56  | 3x3    | 101 us | -        | new    | 1.61 MB |
| 4x96x28x28  | 3x3    | 123 us | -        | new    | 1.20 MB |
| 4x192x14x14 | 3x3    | 122 us | -        | new    | 602 KB  |
| 4x64x56x56  | 3x3 s2 | 126 us | -        | new    | 803 KB  |
| 4x24x56x56  | 7x7    | 181 us | -        | new    | 1.20 MB |
| 4x48x56x56  | 7x7    | 295 us | -        | new    | 2.41 MB |

### Conv1d

| Input      | Kernel | Now     | Apr 2026 | Change   | Mem     |
| ---------- | ------ | ------- | -------- | -------- | ------- |
| 1x16x256   | 3      | 10.2 us | 31.4 us  | **3.1x** | 88.3 KB |
| 8x32x512   | 5      | 1.16 ms | 536 us   | 0.5x     | 1.42 MB |
| 16x64x1024 | 7      | 7.01 ms | 5.17 ms  | 0.7x     | 10.4 MB |

### Depthwise Conv1d

| Input      | Kernel | Now     | Apr 2026 | Change | Mem     |
| ---------- | ------ | ------- | -------- | ------ | ------- |
| 8x32x512   | 3      | 83.0 us | -        | new    | 525 KB  |
| 8x64x1024  | 3 s2   | 108 us  | -        | new    | 1.05 MB |
| 8x64x1024  | 7      | 122 us  | -        | new    | 2.10 MB |
| 4x128x2048 | 15     | 200 us  | -        | new    | 4.19 MB |

---

## Pooling

### Max Pool 2D

| Input        | Kernel | Now    | Apr 2026 | Change   | Mem     |
| ------------ | ------ | ------ | -------- | -------- | ------- |
| 1x64x56x56   | 3x3 s2 | 119 us | 135 us   | **1.1x** | 602 KB  |
| 8x64x56x56   | 3x3 s2 | 559 us | 683 us   | **1.2x** | 4.82 MB |
| 16x128x28x28 | 2x2 s2 | 500 us | 406 us   | 0.8x     | 4.82 MB |
| 1x512x14x14  | 2x2 s2 | 117 us | 90.5 us  | 0.8x     | 301 KB  |

### Max Pool 2D (ResNet)

| Input         | Kernel | Now     | Apr 2026 | Change   | Mem     |
| ------------- | ------ | ------- | -------- | -------- | ------- |
| 1x64x112x112  | 3x3 s2 | 313 us  | 446 us   | **1.4x** | 2.41 MB |
| 8x64x112x112  | 3x3 s2 | 2.08 ms | 2.63 ms  | **1.3x** | 19.3 MB |
| 16x64x112x112 | 3x3 s2 | 3.85 ms | 5.03 ms  | **1.3x** | 38.5 MB |

### Avg Pool 2D

| Input        | Kernel | Now    | Apr 2026 | Change   | Mem     |
| ------------ | ------ | ------ | -------- | -------- | ------- |
| 1x64x56x56   | 3x3 s2 | 113 us | 155 us   | **1.4x** | 201 KB  |
| 8x64x56x56   | 3x3 s2 | 481 us | 782 us   | **1.6x** | 1.60 MB |
| 16x128x28x28 | 2x2 s2 | 470 us | 484 us   | ~1x      | 1.60 MB |

### Adaptive Avg Pool 2D

| Input       | Output | Now     | Apr 2026 | Change   | Mem     |
| ----------- | ------ | ------- | -------- | -------- | ------- |
| 1x256x56x56 | 7x7    | 120 us  | 151 us   | **1.3x** | 50.4 KB |
| 1x512x7x7   | 1x1    | 70.1 us | 63.7 us  | ~1x      | 2.26 KB |
| 8x512x7x7   | 1x1    | 110 us  | 112 us   | ~1x      | 16.6 KB |
| 16x2048x7x7 | 1x1    | 235 us  | 286 us   | **1.2x** | 131 KB  |

### Max Pool 1D

| Input       | Kernel | Now     | Apr 2026 | Change | Mem     |
| ----------- | ------ | ------- | -------- | ------ | ------- |
| 1x64x256    | 3 s2   | 65.9 us | 57.8 us  | 0.9x   | 98.6 KB |
| 8x128x512   | 3 s2   | 330 us  | 316 us   | ~1x    | 3.15 MB |
| 16x256x1024 | 3 s2   | 1.99 ms | 1.73 ms  | 0.9x   | 25.2 MB |

### Kernel Size Comparison (4x64x56x56)

| Kernel | Now    | Apr 2026 | Change   | Mem     |
| ------ | ------ | -------- | -------- | ------- |
| 2x2    | 264 us | 221 us   | 0.8x     | 2.41 MB |
| 3x3    | 306 us | 375 us   | **1.2x** | 2.41 MB |
| 5x5    | 462 us | 1.03 ms  | **2.2x** | 2.41 MB |

---

## Transposed Convolutions

### Conv Transpose 2D

| Input          | Output  | Now     | Apr 2026 | Change   | Mem     |
| -------------- | ------- | ------- | -------- | -------- | ------- |
| 1x64x7x7       | 14x14   | 114 us  | 138 us   | **1.2x** | 398 KB  |
| 1x128x14x14    | 28x28   | 442 us  | 533 us   | **1.2x** | 1.53 MB |
| 1x256x28x28    | 56x56   | 1.41 ms | 2.49 ms  | **1.8x** | 10.1 MB |
| 1x512x7x7 k3s1 | 7x7     | 959 us  | 1.01 ms  | ~1x      | 10.6 MB |
| 8x64x14x14     | 28x28   | 661 us  | 3.41 ms  | **5.2x** | 1.60 MB |
| 1x32x256x256   | 512x512 | 9.58 ms | -        | new      | 42.0 MB |

### DCGAN Generator

| Layer          | Now     | Apr 2026 | Change   | Mem     |
| -------------- | ------- | -------- | -------- | ------- |
| 1x1 to 4x4     | 153 us  | 156 us   | ~1x      | 33.0 KB |
| 4x4 to 8x8     | 141 us  | 234 us   | **1.7x** | 164 KB  |
| 8x8 to 16x16   | 185 us  | 305 us   | **1.6x** | 852 KB  |
| 16x16 to 32x32 | 23.6 us | 26.4 us  | **1.1x** | 193 KB  |

### Conv Transpose 1D

| Input     | Output | Now     | Apr 2026 | Change   | Mem     |
| --------- | ------ | ------- | -------- | -------- | ------- |
| 1x64x32   | 64     | 15.7 us | 17.5 us  | **1.1x** | 49.4 KB |
| 8x128x64  | 128    | 406 us  | 433 us   | ~1x      | 590 KB  |
| 1x256x128 | 256    | 205 us  | 337 us   | **1.6x** | 918 KB  |
| 1x64x2048 | 8192   | 566 us  | -        | new      | 6.42 MB |
| 1x64x3072 | 6144   | 349 us  | -        | new      | 1000 KB |
| 1x32x8192 | 65536  | 1.38 ms | -        | new      | 16.8 MB |

### Conv Transpose 3D

| Input         | Output   | Now     | Apr 2026 | Change   | Mem     |
| ------------- | -------- | ------- | -------- | -------- | ------- |
| 1x32x4x4x4    | 8x8x8    | 152 us  | 245 us   | **1.6x** | 656 KB  |
| 1x64x8x8x8    | 16x16x16 | 557 us  | 1.41 ms  | **2.5x** | 5.24 MB |
| 1x16x32x32x32 | 63x63x63 | 3.58 ms | -        | new      | 24.4 MB |

---

## Interpolation

### Nearest

| Input       | Output  | Now     | Apr 2026 | Change | Mem     |
| ----------- | ------- | ------- | -------- | ------ | ------- |
| 1x3x64x64   | 128x128 | 22.6 us | 22.7 us  | ~1x    | 199 KB  |
| 1x3x32x32   | 128x128 | 22.7 us | 22.8 us  | ~1x    | 199 KB  |
| 1x3x256x256 | 128x128 | 22.3 us | 22.5 us  | ~1x    | 199 KB  |
| 8x3x64x64   | 128x128 | 62.3 us | 58.5 us  | ~1x    | 1.57 MB |
| 1x64x32x32  | 64x64   | 62.8 us | 57.9 us  | ~1x    | 1.05 MB |

### Bilinear

| Input                     | Output  | Now     | Apr 2026 | Change   | Mem     |
| ------------------------- | ------- | ------- | -------- | -------- | ------- |
| 1x3x64x64                 | 128x128 | 67.0 us | 81.2 us  | **1.2x** | 203 KB  |
| 1x3x32x32                 | 128x128 | 69.5 us | 86.5 us  | **1.2x** | 203 KB  |
| 1x3x256x256               | 128x128 | 73.7 us | 83.9 us  | **1.1x** | 203 KB  |
| 8x3x64x64                 | 128x128 | 121 us  | 181 us   | **1.5x** | 1.58 MB |
| 1x64x32x32                | 64x64   | 102 us  | 108 us   | ~1x      | 1.05 MB |
| model, 488x448            | 244x224 | 131 us  | -        | new      | 669 KB  |
| model, 61x56              | 122x112 | 496 us  | -        | new      | 10.5 MB |
| model, 61x56 (half pixel) | 122x112 | 477 us  | -        | new      | 10.5 MB |
| model, 244x224            | 488x448 | 1.74 ms | -        | new      | 42.0 MB |

### Bicubic

| Input       | Output  | Now     | Apr 2026 | Change   | Mem     |
| ----------- | ------- | ------- | -------- | -------- | ------- |
| 1x3x64x64   | 128x128 | 164 us  | 159 us   | ~1x      | 213 KB  |
| 1x3x32x32   | 128x128 | 168 us  | 160 us   | ~1x      | 213 KB  |
| 1x3x256x256 | 128x128 | 163 us  | 159 us   | ~1x      | 213 KB  |
| 8x3x64x64   | 128x128 | 724 us  | 894 us   | **1.2x** | 1.59 MB |
| 1x64x32x32  | 64x64   | 506 us  | 616 us   | **1.2x** | 1.06 MB |
| 1x3x4x4     | 8x8     | 38.0 us | -        | new      | 1.95 KB |

### Lanczos3

| Input       | Output  | Now     | Apr 2026 | Change | Mem     |
| ----------- | ------- | ------- | -------- | ------ | ------- |
| 1x3x64x64   | 128x128 | 166 us  | -        | new    | 223 KB  |
| 1x3x256x256 | 128x128 | 173 us  | -        | new    | 223 KB  |
| 8x3x64x64   | 128x128 | 688 us  | -        | new    | 1.60 MB |
| 1x64x32x32  | 64x64   | 484 us  | -        | new    | 1.06 MB |
| 1x3x4x4     | 8x8     | 30.9 us | -        | new    | 2.59 KB |

---

## Grid Sample 2D

| Input      | Grid  | Now     | Apr 2026 | Change | Mem     |
| ---------- | ----- | ------- | -------- | ------ | ------- |
| 1x3x32x32  | 32x32 | 16.4 us | 16.7 us  | ~1x    | 12.5 KB |
| 1x3x64x64  | 64x64 | 65.3 us | 66.9 us  | ~1x    | 49.4 KB |
| 4x3x32x32  | 32x32 | 65.3 us | 60.7 us  | ~1x    | 49.4 KB |
| 1x16x64x64 | 64x64 | 312 us  | 291 us   | ~1x    | 262 KB  |

---

## Cross Product & Unfold

### Cross Product

| Shape   | Now     | Apr 2026 | Change   | Mem     |
| ------- | ------- | -------- | -------- | ------- |
| 1Kx3    | 20.9 us | 31.1 us  | **1.5x** | 25.2 KB |
| 64Kx3   | 1.21 ms | 1.88 ms  | **1.5x** | 1.57 MB |
| 256Kx3  | 4.87 ms | 7.58 ms  | **1.6x** | 6.29 MB |
| 64x3x64 | 7.60 us | 141 us   | **19x**  | 99.0 KB |

### Unfold (1D)

| Input | Window | Step | Now    | Apr 2026 | Change | Mem    |
| ----- | ------ | ---- | ------ | -------- | ------ | ------ |
| 1K    | 8      | 1    | 158 ns | 62.2 ns  | 0.4x   | 16.8 B |
| 64K   | 8      | 1    | 158 ns | 64.1 ns  | 0.4x   | 16.8 B |
| 64K   | 64     | 1    | 158 ns | 62.2 ns  | 0.4x   | 16.8 B |
| 64K   | 64     | 32   | 158 ns | 62.2 ns  | 0.4x   | 16.8 B |

### Unfold (2D/3D)

| Shape    | Dim | Window | Step | Now    | Apr 2026 | Change | Mem    |
| -------- | --- | ------ | ---- | ------ | -------- | ------ | ------ |
| 256x256  | 1   | 8      | 1    | 160 ns | 68.7 ns  | 0.4x   | 25.2 B |
| 256x256  | 1   | 32     | 16   | 148 ns | 62.5 ns  | 0.4x   | 25.2 B |
| 1024x256 | 1   | 8      | 1    | 148 ns | 62.8 ns  | 0.4x   | 25.2 B |
| 32x64x64 | 2   | 8      | 4    | 157 ns | 77.1 ns  | 0.5x   | 33.8 B |

---

## Deformable Convolutions

### Small/Tiny Inputs

| Input     | Config      | Now     | Apr 2026 | Change | Mem     |
| --------- | ----------- | ------- | -------- | ------ | ------- |
| 1x3x8x8   | 3 to 8, k3  | 9.26 us | 8.72 us  | ~1x    | 14.7 KB |
| 1x3x8x8   | no mask     | 8.42 us | 7.98 us  | ~1x    | 14.7 KB |
| 1x3x16x16 | 3 to 16, k3 | 33.4 us | 36.4 us  | ~1x    | 118 KB  |
| 1x3x16x16 | stride 2    | 9.55 us | 10.2 us  | ~1x    | 20.6 KB |
| 2x8x16x16 | 8 to 16, k3 | 115 us  | 116 us   | ~1x    | 37.7 KB |

### Medium Inputs

| Input      | Config       | Now    | Apr 2026 | Change | Mem     |
| ---------- | ------------ | ------ | -------- | ------ | ------- |
| 1x16x32x32 | 16 to 32, k3 | 823 us | 826 us   | ~1x    | 1.17 MB |
| 1x16x32x32 | wg=4         | 822 us | 840 us   | ~1x    | 390 KB  |
| 1x16x32x32 | og=4         | 931 us | 942 us   | ~1x    | 1.17 MB |

---

## Attention (Scaled Dot-Product)

Flex auto-selects between two gemm-backed strategies:

- **Naive** (score matrix <= 256K elements): Materializes full [seq_q, seq_kv] score matrix. Two
  large gemm calls per (batch, head) amortize dispatch overhead better than many small tiled calls.
- **Flash** (score matrix > 256K elements): Tiles over KV dimension with online softmax.
  `O(seq_q * TILE_KV)` memory per head instead of `O(seq_q * seq_kv)`.

Both fuse scale + softcap + masking + bias + softmax into a single pass.

### Self-Attention

| Config          | Now     | Apr 2026 | Change   | Mem     |
| --------------- | ------- | -------- | -------- | ------- |
| h8, s64, d64    | 94.7 us | 180 us   | **1.9x** | 131 KB  |
| h12, s128, d64  | 216 us  | 989 us   | **4.6x** | 394 KB  |
| h12, s256, d64  | 489 us  | 3.79 ms  | **7.8x** | 787 KB  |
| h12, s512, d64  | 1.50 ms | 14.8 ms  | **9.9x** | 1.57 MB |
| h32, s256, d128 | 1.67 ms | 15.1 ms  | **9.1x** | 4.19 MB |
| b4, h12, s128   | 539 us  | 3.96 ms  | **7.3x** | 1.57 MB |

### Causal Attention

| Config         | Now     | Apr 2026 | Change   | Mem     |
| -------------- | ------- | -------- | -------- | ------- |
| h12, s128, d64 | 216 us  | 1.01 ms  | **4.7x** | 394 KB  |
| h12, s256, d64 | 515 us  | 3.82 ms  | **7.4x** | 787 KB  |
| h12, s512, d64 | 1.51 ms | 14.8 ms  | **9.8x** | 1.57 MB |

### With Additive Bias (ALiBi-style)

| Config         | Now    | Apr 2026 | Change   | Mem    |
| -------------- | ------ | -------- | -------- | ------ |
| h12, s128, d64 | 281 us | 1.04 ms  | **3.7x** | 394 KB |
| h12, s256, d64 | 566 us | 4.00 ms  | **7.1x** | 787 KB |

### Cross-Attention (seq_q != seq_k)

| Config            | Now    | Apr 2026 | Change   | Mem     |
| ----------------- | ------ | -------- | -------- | ------- |
| sq128, sk512, d64 | 525 us | 3.82 ms  | **7.3x** | 394 KB  |
| sq32, sk1024, d64 | 328 us | 2.05 ms  | **6.2x** | 98.6 KB |

### Decode (b1, h8, d64, one query)

| KV length | Now     | Apr 2026 | Change | Mem     |
| --------- | ------- | -------- | ------ | ------- |
| 1         | 880 ns  | -        | new    | 2.34 KB |
| 64        | 6.39 us | -        | new    | 2.50 KB |
| 128       | 11.6 us | -        | new    | 2.75 KB |
| 256       | 44.7 us | -        | new    | 2.34 KB |
| 512       | 44.2 us | -        | new    | 2.34 KB |

### Transformer Shapes (h32, 8 KV heads, d128; f16, training in f32)

| Config            | Now     | Apr 2026 | Change | Mem     |
| ----------------- | ------- | -------- | ------ | ------- |
| prefill b1, s512  | 7.64 ms | -        | new    | 21.0 MB |
| decode b8, s4096  | 25.2 ms | -        | new    | 269 MB  |
| training b4, s512 | 20.3 ms | -        | new    | 33.5 MB |

---

## Quantized Tensor Operations

All quantized ops (except layout ops) go through a dequantize-op-quantize cycle. Flex stores scales
separately and applies `scale * x_q` directly.

### Quantize (float to i8)

| Size | Now     | Apr 2026 | Change   | Mem     |
| ---- | ------- | -------- | -------- | ------- |
| 4K   | 2.32 us | 6.93 us  | **3.0x** | 12.4 KB |
| 64K  | 30.6 us | 109 us   | **3.6x** | 328 KB  |
| 1M   | 490 us  | 1.75 ms  | **3.6x** | 5.24 MB |

### Dequantize (i8 to float)

| Size | Now     | Apr 2026 | Change | Mem     |
| ---- | ------- | -------- | ------ | ------- |
| 4K   | 464 ns  | 399 ns   | 0.9x   | 16.4 KB |
| 64K  | 3.73 us | 3.73 us  | ~1x    | 262 KB  |
| 1M   | 50.4 us | 54.6 us  | ~1x    | 4.19 MB |

### q_add (dequant + add + requant)

| Size | Now     | Apr 2026 | Change | Mem     |
| ---- | ------- | -------- | ------ | ------- |
| 4K   | 1.26 us | 1.15 us  | ~1x    | 20.6 KB |
| 64K  | 14.2 us | 14.3 us  | ~1x    | 524 KB  |
| 1M   | 205 us  | 208 us   | ~1x    | 8.39 MB |

### q_matmul (dequant + matmul + requant)

| Size    | Now     | Apr 2026 | Change | Mem     |
| ------- | ------- | -------- | ------ | ------- |
| 64x64   | 7.10 us | 7.10 us  | ~1x    | 66.5 KB |
| 256x256 | 158 us  | 147 us   | ~1x    | 1.05 MB |
| 512x512 | 618 us  | 641 us   | ~1x    | 4.19 MB |

### q_sum (dequant + sum)

| Size | Now     | Apr 2026 | Change   | Mem     |
| ---- | ------- | -------- | -------- | ------- |
| 4K   | 648 ns  | 605 ns   | ~1x      | 2.13 KB |
| 64K  | 7.29 us | 7.71 us  | ~1x      | 262 KB  |
| 1M   | 93.9 us | 194 us   | **2.1x** | 4.19 MB |

### q_permute (zero-copy layout op)

| Size      | Now    | Apr 2026 | Change | Mem    |
| --------- | ------ | -------- | ------ | ------ |
| 256x256   | 148 ns | 75.5 ns  | 0.5x   | 21.0 B |
| 1024x1024 | 154 ns | 77.5 ns  | 0.5x   | 21.0 B |

### q_argmax

| Size      | Now     | Apr 2026 | Change | Mem     |
| --------- | ------- | -------- | ------ | ------- |
| 256x256   | 76.6 us | 67.9 us  | 0.9x   | 3.25 KB |
| 1024x1024 | 171 us  | 137 us   | 0.8x   | 12.5 KB |

### q_argmin

| Size      | Now     | Apr 2026 | Change | Mem     |
| --------- | ------- | -------- | ------ | ------- |
| 256x256   | 75.8 us | 69.4 us  | ~1x    | 3.25 KB |
| 1024x1024 | 143 us  | 139 us   | ~1x    | 12.5 KB |

### q_gather

| Size      | Now     | Apr 2026 | Change | Mem     |
| --------- | ------- | -------- | ------ | ------- |
| 256x256   | 64.8 us | 65.7 us  | ~1x    | 590 KB  |
| 1024x1024 | 375 us  | 393 us   | ~1x    | 9.44 MB |

---

## Default Ops (sort, repeat, creation, embedding, predicates)

These ops override burn's default trait implementations with direct storage operations.

### Sort (f32, 1D)

| Size | Now     | Apr 2026 | Change | Mem     |
| ---- | ------- | -------- | ------ | ------- |
| 4K   | 53.4 us | 53.6 us  | ~1x    | 16.5 KB |
| 64K  | 595 us  | 593 us   | ~1x    | 262 KB  |
| 1M   | 8.43 ms | 8.47 ms  | ~1x    | 4.19 MB |

### Sort (f32, 2D along last dim)

| Size      | Now     | Apr 2026 | Change | Mem     |
| --------- | ------- | -------- | ------ | ------- |
| 64x64     | 5.82 us | 5.72 us  | ~1x    | 16.5 KB |
| 256x256   | 198 us  | 200 us   | ~1x    | 262 KB  |
| 1024x1024 | 1.14 ms | 1.17 ms  | ~1x    | 4.19 MB |

### Argsort (f32, 1D)

| Size | Now     | Apr 2026 | Change | Mem     |
| ---- | ------- | -------- | ------ | ------- |
| 4K   | 71.5 us | 70.2 us  | ~1x    | 65.5 KB |
| 1M   | 12.7 ms | 12.8 ms  | ~1x    | 16.8 MB |

### Repeat Dim (f32, 256x256)

| Config            | Now     | Apr 2026 | Change | Mem     |
| ----------------- | ------- | -------- | ------ | ------- |
| dim0 4x           | 17.9 us | 12.7 us  | 0.7x   | 1.05 MB |
| dim1 4x           | 18.4 us | 11.5 us  | 0.6x   | 1.05 MB |
| dim0 8x (512x512) | 134 us  | 98.4 us  | 0.7x   | 8.39 MB |

### Tensor Creation (f32, 1M elements)

| Operation | Now     | Apr 2026 | Change | Mem     |
| --------- | ------- | -------- | ------ | ------- |
| zeros     | 16.5 us | 17.7 us  | ~1x    | 4.19 MB |
| ones      | 33.5 us | 35.7 us  | ~1x    | 4.19 MB |
| full      | 33.8 us | 35.9 us  | ~1x    | 4.19 MB |

### Arange

| Size | Now     | Apr 2026 | Change | Mem     |
| ---- | ------- | -------- | ------ | ------- |
| 4K   | 1.27 us | 1.21 us  | ~1x    | 16.4 KB |
| 1M   | 273 us  | 299 us   | ~1x    | 4.19 MB |

### Embedding (f32)

| Config                  | Now     | Apr 2026 | Change | Mem     |
| ----------------------- | ------- | -------- | ------ | ------- |
| 30k vocab, d=512, 8x128 | 26.3 us | 26.1 us  | ~1x    | 2.10 MB |
| 50k vocab, d=768, 4x256 | 39.1 us | 38.9 us  | ~1x    | 3.15 MB |

### Predicates (f32, 1M elements)

| Operation | Now     | Apr 2026 | Change | Mem     |
| --------- | ------- | -------- | ------ | ------- |
| is_nan    | 46.5 us | 46.4 us  | ~1x    | 1.05 MB |
| is_inf    | 49.3 us | 52.6 us  | ~1x    | 1.05 MB |

---

## FFT (Real FFT)

From `burn-signal`, which enables Flex's native FFT kernels. Measured at `c800a9e5e`, which has the
same Flex and tensor code as the commit above.

### 1D rfft

| Size    | Now     | Apr 2026 | Change | Mem     |
| ------- | ------- | -------- | ------ | ------- |
| n=256   | 986 ns  | 841 ns   | 0.9x   | 1.70 KB |
| n=1024  | 2.87 us | 2.64 us  | ~1x    | 8.36 KB |
| n=4096  | 10.8 us | 10.6 us  | ~1x    | 49.3 KB |
| n=16384 | 46.8 us | 45.4 us  | ~1x    | 197 KB  |
| n=65536 | 236 us  | 231 us   | ~1x    | 787 KB  |

### Batched 2D rfft (along last dim)

| Size      | Now     | Apr 2026 | Change   | Mem     |
| --------- | ------- | -------- | -------- | ------- |
| 16 x 1024 | 61.3 us | 59.9 us  | ~1x      | 66.6 KB |
| 64 x 1024 | 118 us  | 114 us   | ~1x      | 266 KB  |
| 256 x 256 | 149 us  | 191 us   | **1.3x** | 278 KB  |

### 1D irfft (inverse)

| Size    | Now     | Apr 2026 | Change   | Mem     |
| ------- | ------- | -------- | -------- | ------- |
| n=256   | 1.25 us | 1.32 us  | ~1x      | 4.22 KB |
| n=1024  | 3.50 us | 4.01 us  | **1.1x** | 16.5 KB |
| n=4096  | 13.4 us | 15.6 us  | **1.2x** | 65.7 KB |
| n=16384 | 54.6 us | 63.2 us  | **1.2x** | 262 KB  |
| n=65536 | 268 us  | 278 us   | ~1x      | 1.05 MB |

Flex implementation: Cooley-Tukey with mixed radix-4/radix-2, complex packing (forward) / inverse
packing (inverse), compile-time twiddle tables via const fn, SIMD vectorization via macerator,
unrolled small kernels (N=2,4,8), and rayon parallelism across fibers.

---

## Running Benchmarks

```bash
cargo bench -p burn-backend-tests --features flex --bench attention
cargo bench -p burn-backend-tests --features flex --bench binary_ops
cargo bench -p burn-backend-tests --features flex --bench cat_max_min_ops
cargo bench -p burn-backend-tests --features flex --bench comparison_ops
cargo bench -p burn-backend-tests --features flex --bench conv_ops
cargo bench -p burn-backend-tests --features flex --bench conv_transpose_ops
cargo bench -p burn-backend-tests --features flex --bench cross_unfold_ops
cargo bench -p burn-backend-tests --features flex --bench cumulative_ops
cargo bench -p burn-backend-tests --features flex --bench default_ops
cargo bench -p burn-backend-tests --features flex --bench deform_conv_ops
cargo bench -p burn-backend-tests --features flex --bench gather_scatter_ops
cargo bench -p burn-backend-tests --features flex --bench int_ops
cargo bench -p burn-backend-tests --features flex --bench interpolate_ops
cargo bench -p burn-backend-tests --features flex --bench matmul
cargo bench -p burn-backend-tests --features flex --bench norm_ops
cargo bench -p burn-backend-tests --features flex --bench pool_ops
cargo bench -p burn-backend-tests --features flex --bench quantization_ops
cargo bench -p burn-backend-tests --features flex --bench reduce_ops
cargo bench -p burn-backend-tests --features flex --bench slice_ops
cargo bench -p burn-backend-tests --features flex --bench softmax_ops
cargo bench -p burn-backend-tests --features flex --bench unary_ops
cargo bench -p burn-signal --no-default-features --features std,flex,burn-flex/default --bench fft_ops
```
