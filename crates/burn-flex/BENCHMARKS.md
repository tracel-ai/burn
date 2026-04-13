# Benchmarks: Flex vs NdArray

All benchmarks run on Apple M3 Max, comparing burn-flex against burn-ndarray. Default features
enabled (`std`, `simd`, `rayon`); `gemm` is a required dependency.

**Date**: 2026-04-06

## How to Read

- **Median** time reported (lower is better)
- **Speedup** = NdArray median / Flex median
- **Mem** = peak allocation (`max alloc` from divan)
- Bold speedup means Flex wins; plain means tie or NdArray wins

---

## Binary Operations (f32)

| Operation   | Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ----------- | ---- | ------- | ------- | -------- | -------- | ----------- |
| add         | 4K   | 389 ns  | 436 ns  | **1.1x** | 16.4 KB  | 16.4 KB     |
| add         | 64K  | 7.36 us | 7.45 us | ~1x      | 262 KB   | 262 KB      |
| add         | 1M   | 83.9 us | 115 us  | **1.4x** | 4.19 MB  | 4.19 MB     |
| mul         | 4K   | 382 ns  | 430 ns  | **1.1x** | 16.4 KB  | 16.4 KB     |
| mul         | 64K  | 7.40 us | 7.40 us | ~1x      | 262 KB   | 262 KB      |
| mul         | 1M   | 115 us  | 115 us  | ~1x      | 4.19 MB  | 4.19 MB     |
| div         | 1M   | 115 us  | 115 us  | ~1x      | 4.19 MB  | 4.19 MB     |
| add_scalar  | 1M   | 78.7 us | 87.8 us | **1.1x** | 4.19 MB  | 4.19 MB     |
| mul_scalar  | 1M   | 75.8 us | 87.5 us | **1.2x** | 4.19 MB  | 4.19 MB     |
| powf        | 64K  | 197 us  | 199 us  | ~1x      | 262 KB   | 262 KB      |
| powf        | 1M   | 3.17 ms | 3.21 ms | ~1x      | 4.19 MB  | 4.19 MB     |
| powf_scalar | 1M   | 3.23 ms | 3.18 ms | ~1x      | 4.19 MB  | 4.19 MB     |
| atan2       | 64K  | 143 us  | 142 us  | ~1x      | 262 KB   | 262 KB      |
| atan2       | 1M   | 2.33 ms | 2.32 ms | ~1x      | 4.19 MB  | 4.19 MB     |

### Transposed

| Operation | Size      | Flex    | NdArray | Speedup | Flex Mem | NdArray Mem |
| --------- | --------- | ------- | ------- | ------- | -------- | ----------- |
| add       | 256x256   | 48.5 us | 46.0 us | 0.95x   | 262 KB   | 262 KB      |
| add       | 1024x1024 | 1.00 ms | 990 us  | ~1x     | 4.19 MB  | 4.19 MB     |

---

## Binary Operations (i64)

| Operation      | Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| -------------- | ---- | ------- | ------- | -------- | -------- | ----------- |
| int_add        | 4K   | 361 ns  | 655 ns  | **1.8x** | 16.5 KB  | 32.8 KB     |
| int_add        | 64K  | 7.40 us | 14.6 us | **2.0x** | 262 KB   | 524 KB      |
| int_add        | 1M   | 115 us  | 230 us  | **2.0x** | 4.19 MB  | 8.39 MB     |
| int_mul        | 4K   | 366 ns  | 1.95 us | **5.3x** | 16.4 KB  | 32.8 KB     |
| int_mul        | 64K  | 7.40 us | 26.7 us | **3.6x** | 262 KB   | 524 KB      |
| int_mul        | 1M   | 115 us  | 230 us  | **2.0x** | 4.19 MB  | 8.39 MB     |
| int_div        | 1M   | 604 us  | 698 us  | **1.2x** | 4.19 MB  | 8.39 MB     |
| int_add_scalar | 1M   | 75.8 us | 174 us  | **2.3x** | 4.19 MB  | 8.39 MB     |
| int_mul_scalar | 1M   | 75.7 us | 258 us  | **3.4x** | 4.19 MB  | 8.39 MB     |

### Int Power

| Operation | Size     | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | -------- | ------- | ------- | -------- | -------- | ----------- |
| int_powi  | 256x256  | 95.6 us | 83.1 us | 0.87x    | 262 KB   | 524 KB      |
| int_powi  | 1024x256 | 336 us  | 382 us  | **1.1x** | 1.05 MB  | 2.10 MB     |

### Transposed (i64)

| Operation | Size      | Flex    | NdArray | Speedup  |
| --------- | --------- | ------- | ------- | -------- |
| int_add   | 256x256   | 55.6 us | 50.5 us | 0.91x    |
| int_add   | 1024x1024 | 996 us  | 1.10 ms | **1.1x** |

---

## Int Cast

| Operation  | Size      | Flex    | NdArray | Speedup     | Flex Mem | NdArray Mem |
| ---------- | --------- | ------- | ------- | ----------- | -------- | ----------- |
| i64 to i8  | 256x256   | 3.19 us | 20.2 us | **6.3x**    | 65.6 KB  | 65.6 KB     |
| i64 to i32 | 64x64     | 16.8 ns | 1.37 us | **82x**     | 16.0 B   | 16.4 KB     |
| i64 to i32 | 256x256   | 13.6 ns | 20.0 us | **~1475x**  | 16.0 B   | 262 KB      |
| i64 to i32 | 1024x1024 | 13.6 ns | 352 us  | **~25963x** | 16.0 B   | 4.19 MB     |

---

## Int Random

| Operation | Size       | Flex    | NdArray | Speedup  |
| --------- | ---------- | ------- | ------- | -------- |
| uniform   | 64x64      | 20.9 us | 31.6 us | **1.5x** |
| uniform   | 256x256    | 334 us  | 510 us  | **1.5x** |
| uniform   | 1024x1024  | 5.36 ms | 8.16 ms | **1.5x** |
| uniform   | 16x128x128 | 1.35 ms | 2.04 ms | **1.5x** |

---

## Matrix Multiplication

### Square (f32)

| Size      | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ------- | ------- | -------- | -------- | ----------- |
| 64x64     | 6.06 us | 18.9 us | **3.1x** | 33.6 KB  | 49.3 KB     |
| 128x128   | 43.8 us | 41.8 us | ~1x      | 328 KB   | 197 KB      |
| 256x256   | 166 us  | 138 us  | 0.83x    | 524 KB   | 786 KB      |
| 512x512   | 579 us  | 840 us  | **1.4x** | 2.10 MB  | 3.15 MB     |
| 1024x1024 | 2.69 ms | 5.83 ms | **2.2x** | 8.39 MB  | 12.6 MB     |

### Rectangular (f32)

| Shape               | Flex   | NdArray | Speedup |
| ------------------- | ------ | ------- | ------- |
| 512x64 x 64x512     | 167 us | 144 us  | 0.87x   |
| 256x512 x 512x256   | 265 us | 266 us  | ~1x     |
| 128x1024 x 1024x128 | 190 us | 199 us  | ~1x     |

### Transposed (256x256)

| Config          | Flex   | NdArray | Speedup  |
| --------------- | ------ | ------- | -------- |
| LHS transposed  | 140 us | 174 us  | **1.2x** |
| RHS transposed  | 158 us | 173 us  | **1.1x** |
| Both transposed | 165 us | 210 us  | **1.3x** |

### Batched (f32)

| Shape              | Flex    | NdArray | Speedup  |
| ------------------ | ------- | ------- | -------- |
| 8x 64x64           | 57.6 us | 76.7 us | **1.3x** |
| 32x 64x64          | 67.0 us | 111 us  | **1.7x** |
| 16x 128x128        | 267 us  | 540 us  | **2.0x** |
| 12x 512x64 (heads) | 777 us  | 1.71 ms | **2.2x** |

### Broadcast (f32)

| Shape                     | Flex    | NdArray | Speedup  |
| ------------------------- | ------- | ------- | -------- |
| [1,64,64] x [8,64,64]     | 47.9 us | 79.8 us | **1.7x** |
| [8,64,64] x [1,64,64]     | 54.0 us | 78.5 us | **1.5x** |
| [2,1,32,32] x [1,4,32,32] | 7.00 us | 40.0 us | **5.7x** |
| [4,1,64,64] x [1,4,64,64] | 50.0 us | 66.4 us | **1.3x** |

### Integer (i32)

| Size    | Flex    | NdArray | Speedup  |
| ------- | ------- | ------- | -------- |
| 64x64   | 30.2 us | 110 us  | **3.7x** |
| 128x128 | 196 us  | 971 us  | **4.9x** |
| 256x256 | 1.90 ms | 10.1 ms | **5.3x** |
| 512x512 | 18.3 ms | 119 ms  | **6.5x** |

---

## Slice Operations

### Basic Slicing

| Operation | Size      | Flex   | NdArray | Speedup   | Flex Mem | NdArray Mem |
| --------- | --------- | ------ | ------- | --------- | -------- | ----------- |
| slice 1D  | 1K        | 112 ns | 235 ns  | **2.1x**  | 56.0 B   | 2.15 KB     |
| slice 1D  | 1M        | 105 ns | 26.4 us | **~252x** | 8.75 B   | 2.10 MB     |
| slice 2D  | 256x256   | 120 ns | 3.61 us | **30x**   | 19.0 B   | 65.7 KB     |
| slice 2D  | 1024x1024 | 115 ns | 31.9 us | **~278x** | 17.5 B   | 1.05 MB     |
| slice 3D  | 64x64x64  | 148 ns | 16.0 us | **~108x** | 28.5 B   | 131 KB      |

### Narrow

| Operation   | Size      | Flex   | NdArray | Speedup   |
| ----------- | --------- | ------ | ------- | --------- |
| narrow dim0 | 256x256   | 141 ns | 1.70 us | **12x**   |
| narrow dim0 | 1024x1024 | 142 ns | 26.0 us | **~183x** |
| narrow dim1 | 256x256   | 128 ns | 6.84 us | **53x**   |

### Slice Assignment

| Operation | Size      | Flex    | NdArray | Speedup  |
| --------- | --------- | ------- | ------- | -------- |
| assign 1D | 1K        | 304 ns  | 395 ns  | **1.3x** |
| assign 2D | 256x256   | 5.39 us | 5.69 us | **1.1x** |
| assign 2D | 1024x1024 | 74.9 us | 74.8 us | ~1x      |

### Transposed Slicing

| Size      | Flex    | NdArray | Speedup    |
| --------- | ------- | ------- | ---------- |
| 256x256   | 98.2 ns | 7.98 us | **81x**    |
| 1024x1024 | 98.2 ns | 232 us  | **~2363x** |

### Slice with Step

| Operation | Size      | Flex    | NdArray | Speedup    |
| --------- | --------- | ------- | ------- | ---------- |
| step2 1D  | 1K        | 87.1 ns | 360 ns  | **4.1x**   |
| step2 1D  | 1M        | 76.7 ns | 142 us  | **~1849x** |
| step2 2D  | 1024x1024 | 103 ns  | 86.8 us | **~845x**  |
| step4 2D  | 256x256   | 101 ns  | 2.65 us | **26x**    |

---

## Concatenation

### Cat (dim 0, contiguous memcpy fast path)

| Tensors | Size     | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ------- | -------- | ------- | ------- | -------- | -------- | ----------- |
| 4x      | 256x256  | 16.0 us | 33.1 us | **2.1x** | 1.05 MB  | 2.10 MB     |
| 4x      | 1024x256 | 57.9 us | 132 us  | **2.3x** | 4.20 MB  | 8.39 MB     |
| 16x     | 64x64    | 4.11 us | 11.5 us | **2.8x** | 265 KB   | 528 KB      |
| 4x      | 16K (1D) | 3.47 us | 10.1 us | **2.9x** | 263 KB   | 525 KB      |

### Cat (dim 1, general path)

| Tensors | Size    | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ------- | ------- | ------- | ------- | -------- | -------- | ----------- |
| 4x      | 256x64  | 6.92 us | 59.2 us | **8.6x** | 263 KB   | 525 KB      |
| 4x      | 1024x64 | 25.5 us | 366 us  | **14x**  | 1.05 MB  | 2.10 MB     |

Dim-1 cat is much faster because NdArray's default uses N `slice_assign` calls while Flex copies
contiguous chunks directly.

---

## Reduce Operations

### Full Tensor Sum

| Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | -------- | -------- | ----------- |
| 1K   | 118 ns  | 156 ns  | **1.3x** | 76.2 B   | 44.0 B      |
| 64K  | 3.23 us | 6.20 us | **1.9x** | 80.0 B   | 44.0 B      |
| 1M   | 43.3 us | 97.0 us | **2.2x** | 84.0 B   | 44.0 B      |

### Full Tensor Max

| Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | -------- | -------- | ----------- |
| 1K   | 207 ns  | 662 ns  | **3.2x** | 76.2 B   | 44.0 B      |
| 64K  | 8.78 us | 32.5 us | **3.7x** | 84.0 B   | 44.0 B      |
| 1M   | 139 us  | 558 us  | **4.0x** | 84.0 B   | 44.0 B      |

### Full Tensor Min

| Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | -------- | -------- | ----------- |
| 1K   | 278 ns  | 579 ns  | **2.1x** | 84.0 B   | 44.0 B      |
| 64K  | 9.15 us | 34.8 us | **3.8x** | 84.0 B   | 44.0 B      |
| 1M   | 142 us  | 540 us  | **3.8x** | 84.0 B   | 44.0 B      |

### Int Max

| Size      | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 2.84 us | 9.12 us | **3.2x** | 84.0 B   | 48.0 B      |
| 1024x1024 | 42.2 us | 145 us  | **3.4x** | 92.0 B   | 48.0 B      |

### Sum Along Dimension

| Shape     | Dim | Flex    | NdArray | Speedup  |
| --------- | --- | ------- | ------- | -------- |
| 256x256   | 0   | 5.06 us | 11.4 us | **2.3x** |
| 256x256   | 1   | 2.77 us | 4.61 us | **1.7x** |
| 1024x1024 | 0   | 80.0 us | 100 us  | **1.3x** |
| 1024x1024 | 1   | 42.2 us | 82.0 us | **1.9x** |

### 3D Sum (Batched)

| Shape      | Dim | Flex    | NdArray | Speedup  |
| ---------- | --- | ------- | ------- | -------- |
| 32x256x256 | 1   | 156 us  | 212 us  | **1.4x** |
| 32x256x256 | 2   | 86.2 us | 134 us  | **1.6x** |

### Sum Transposed

| Size      | Flex    | NdArray | Speedup  |
| --------- | ------- | ------- | -------- |
| 256x256   | 3.27 us | 6.20 us | **1.9x** |
| 1024x1024 | 41.4 us | 96.7 us | **2.3x** |

### Sum Dim on Transposed

| Size      | Dim | Flex    | NdArray | Speedup  |
| --------- | --- | ------- | ------- | -------- |
| 256x256   | 0   | 2.79 us | 4.40 us | **1.6x** |
| 1024x1024 | 0   | 41.5 us | 81.9 us | **2.0x** |

### Mean Along Dimension

| Shape     | Dim | Flex    | NdArray | Speedup  |
| --------- | --- | ------- | ------- | -------- |
| 256x256   | 1   | 2.90 us | 4.53 us | **1.6x** |
| 1024x1024 | 1   | 42.5 us | 82.4 us | **1.9x** |

### Argmax

| Shape     | Dim | Flex    | NdArray | Speedup  |
| --------- | --- | ------- | ------- | -------- |
| 1K        | -   | 752 ns  | 4.09 us | **5.4x** |
| 256x256   | 1   | 66.7 us | 242 us  | **3.6x** |
| 1024x1024 | 1   | 120 us  | 3.98 ms | **33x**  |

---

## Cumulative Operations

### Cumsum

| Shape     | Dim | Flex    | NdArray | Speedup  |
| --------- | --- | ------- | ------- | -------- |
| 1K        | 0   | 838 ns  | 65.7 us | **78x**  |
| 64K       | 0   | 45.2 us | 4.25 ms | **94x**  |
| 1M        | 0   | 719 us  | 68.1 ms | **95x**  |
| 256x256   | 0   | 11.3 us | 34.2 us | **3.0x** |
| 256x256   | 1   | 42.6 us | 215 us  | **5.0x** |
| 1024x1024 | 1   | 709 us  | 5.51 ms | **7.8x** |

### Cumprod

| Shape   | Dim | Flex    | NdArray | Speedup  |
| ------- | --- | ------- | ------- | -------- |
| 1K      | 0   | 1.27 us | 66.0 us | **52x**  |
| 256x256 | 1   | 66.3 us | 216 us  | **3.3x** |

### Cummin

| Shape     | Dim | Flex    | NdArray | Speedup  |
| --------- | --- | ------- | ------- | -------- |
| 1K        | 0   | 1.79 us | 66.3 us | **37x**  |
| 256x256   | 1   | 102 us  | 204 us  | **2.0x** |
| 1024x1024 | 1   | 1.71 ms | 5.53 ms | **3.2x** |

### Cummax

| Shape     | Dim | Flex    | NdArray | Speedup  |
| --------- | --- | ------- | ------- | -------- |
| 1K        | 0   | 1.82 us | 65.9 us | **36x**  |
| 256x256   | 1   | 102 us  | 123 us  | **1.2x** |
| 1024x1024 | 1   | 1.70 ms | 3.60 ms | **2.1x** |

### 3D Cumsum (Batched)

| Shape    | Dim | Flex    | NdArray | Speedup  |
| -------- | --- | ------- | ------- | -------- |
| 32x64x64 | 1   | 24.1 us | 83.7 us | **3.5x** |
| 32x64x64 | 2   | 68.0 us | 237 us  | **3.5x** |

---

## Gather/Scatter Operations

### Gather

| Shape     | Dim | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | --- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 0   | 32.9 us | 140 us  | **4.3x** | 393 KB   | 786 KB      |
| 256x256   | 1   | 33.8 us | 87.1 us | **2.6x** | 393 KB   | 786 KB      |
| 1024x1024 | 1   | 273 us  | 1.31 ms | **4.8x** | 6.29 MB  | 12.6 MB     |

### Scatter Add

| Shape     | Dim | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | --- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 1   | 35.7 us | 189 us  | **5.3x** | 524 KB   | 918 KB      |
| 1024x1024 | 1   | 563 us  | 2.83 ms | **5.0x** | 8.39 MB  | 14.7 MB     |

### Select

| Shape     | Dim | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | --- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 0   | 2.02 us | 12.9 us | **6.4x** | 132 KB   | 143 KB      |
| 256x256   | 1   | 26.3 us | 31.1 us | **1.2x** | 132 KB   | 143 KB      |
| 1024x1024 | 0   | 26.8 us | 88.1 us | **3.3x** | 2.10 MB  | 2.15 MB     |

### Bool Select

| Shape    | Indices | Flex    | NdArray | Speedup | Flex Mem | NdArray Mem |
| -------- | ------- | ------- | ------- | ------- | -------- | ----------- |
| 256x256  | 128     | 935 ns  | 12.0 us | **13x** | 33.9 KB  | 45.0 KB     |
| 1024x256 | 512     | 2.89 us | 47.4 us | **16x** | 135 KB   | 180 KB      |

### Select Add

| Shape     | Dim | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | --- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 0   | 7.35 us | 13.5 us | **1.8x** | 263 KB   | 263 KB      |
| 1024x1024 | 0   | 103 us  | 126 us  | **1.2x** | 4.20 MB  | 4.20 MB     |

---

## Unary Operations

### Basic Math

| Operation | Size | Flex    | NdArray | Speedup  |
| --------- | ---- | ------- | ------- | -------- |
| exp       | 4K   | 5.07 us | 5.20 us | ~1x      |
| exp       | 64K  | 80.5 us | 85.0 us | **1.1x** |
| exp       | 1M   | 1.31 ms | 1.35 ms | ~1x      |
| log       | 4K   | 6.74 us | 6.87 us | ~1x      |
| log       | 64K  | 106 us  | 111 us  | ~1x      |
| log       | 1M   | 1.72 ms | 1.77 ms | ~1x      |
| sqrt      | 4K   | 612 ns  | 860 ns  | **1.4x** |
| sqrt      | 64K  | 9.03 us | 12.8 us | **1.4x** |
| sqrt      | 1M   | 142 us  | 195 us  | **1.4x** |
| abs       | 1M   | 75.8 us | 75.8 us | ~1x      |
| recip     | 1M   | 75.5 us | 75.6 us | ~1x      |

### Trigonometric

| Operation | Size | Flex    | NdArray | Speedup  |
| --------- | ---- | ------- | ------- | -------- |
| sin       | 4K   | 5.65 us | 8.04 us | **1.4x** |
| sin       | 64K  | 89.2 us | 130 us  | **1.5x** |
| sin       | 1M   | 1.45 ms | 2.10 ms | **1.4x** |
| cos       | 4K   | 6.57 us | 8.45 us | **1.3x** |
| cos       | 1M   | 1.68 ms | 2.21 ms | **1.3x** |
| tanh      | 4K   | 7.07 us | 13.7 us | **1.9x** |
| tanh      | 64K  | 112 us  | 222 us  | **2.0x** |
| tanh      | 1M   | 1.80 ms | 3.57 ms | **2.0x** |

### Transposed (Non-contiguous)

| Operation | Size      | Flex    | NdArray | Speedup  |
| --------- | --------- | ------- | ------- | -------- |
| exp       | 256x256   | 80.1 us | 84.8 us | **1.1x** |
| exp       | 1024x1024 | 1.31 ms | 1.35 ms | ~1x      |

---

## Comparison & Boolean Operations

### Tensor-Tensor Comparisons

| Operation | Size | Flex    | NdArray | Speedup | Flex Mem | NdArray Mem |
| --------- | ---- | ------- | ------- | ------- | -------- | ----------- |
| greater   | 4K   | 431 ns  | 398 ns  | ~1x     | 4.17 KB  | 4.14 KB     |
| greater   | 64K  | 6.48 us | 5.73 us | ~1x     | 65.6 KB  | 65.6 KB     |
| greater   | 1M   | 93 us   | 88 us   | ~1x     | 1.05 MB  | 1.05 MB     |
| equal     | 4K   | 433 ns  | 403 ns  | ~1x     | 4.17 KB  | 4.14 KB     |
| equal     | 1M   | 86 us   | 89 us   | ~1x     | 1.05 MB  | 1.05 MB     |
| lower     | 1M   | 92 us   | 87 us   | ~1x     | 1.05 MB  | 1.05 MB     |

### Scalar Comparisons

| Operation    | Size | Flex  | NdArray | Speedup   |
| ------------ | ---- | ----- | ------- | --------- |
| greater_elem | 1M   | 56 us | 76 us   | **1.36x** |

### Transposed Comparisons

| Operation | Size      | Flex    | NdArray | Speedup |
| --------- | --------- | ------- | ------- | ------- |
| greater   | 256x256   | 53.6 us | 43.7 us | 0.82x   |
| greater   | 1024x1024 | 985 us  | 990 us  | ~1x     |

### Broadcast Comparisons

| Operation | Shape     | Flex    | NdArray | Speedup  |
| --------- | --------- | ------- | ------- | -------- |
| greater   | 256x256   | 7.98 us | 25.6 us | **3.2x** |
| greater   | 1024x1024 | 120 us  | 317 us  | **2.6x** |

### Expand (Broadcasting)

| Operation           | Flex   | NdArray | Speedup    |
| ------------------- | ------ | ------- | ---------- |
| 1x1 to 1000x1000    | 126 ns | 291 us  | **~2307x** |
| 1024x1 to 1024x1024 | 110 ns | 310 us  | **~2803x** |
| 1x1024 to 1024x1024 | 126 ns | 78.6 us | **~623x**  |

### Boolean Operations

| Operation | Size | Flex    | NdArray | Speedup |
| --------- | ---- | ------- | ------- | ------- |
| bool_not  | 1M   | 24.1 us | 19.0 us | 0.79x   |
| bool_and  | 1M   | 34.6 us | 28.8 us | 0.83x   |

---

## Convolutions

### Kernel Size Comparison (4x64x56x56, 64 to 128 channels)

| Kernel | Flex    | NdArray | Speedup  |
| ------ | ------- | ------- | -------- |
| 1x1    | 577 us  | 803 us  | **1.4x** |
| 3x3    | 3.65 ms | 9.46 ms | **2.6x** |
| 5x5    | 8.05 ms | 24.7 ms | **3.1x** |
| 7x7    | 15.7 ms | 49.8 ms | **3.2x** |

### ResNet Layers (batch=1, 3x3)

| Layer  | Input       | Channels       | Flex    | NdArray | Speedup  |
| ------ | ----------- | -------------- | ------- | ------- | -------- |
| conv1  | 1x3x224x224 | 3 to 64 (k7s2) | 954 us  | 1.26 ms | **1.3x** |
| layer1 | 1x64x56x56  | 64 to 64       | 986 us  | 1.82 ms | **1.8x** |
| layer2 | 1x128x28x28 | 128 to 128     | 1.08 ms | 1.60 ms | **1.5x** |
| layer3 | 1x256x14x14 | 256 to 256     | 1.65 ms | 3.08 ms | **1.9x** |
| layer4 | 1x512x7x7   | 512 to 512     | 2.71 ms | 10.3 ms | **3.8x** |

### Small (batch=1, 3x3)

| Input      | Channels | Flex    | NdArray | Speedup  |
| ---------- | -------- | ------- | ------- | -------- |
| 1x3x32x32  | 3 to 16  | 71.3 us | 79.4 us | **1.1x** |
| 1x16x32x32 | 16 to 32 | 219 us  | 252 us  | **1.2x** |
| 1x32x16x16 | 32 to 64 | 164 us  | 338 us  | **2.1x** |

### Large Batched (batch=16, 3x3)

| Input         | Channels   | Flex    | NdArray | Speedup  |
| ------------- | ---------- | ------- | ------- | -------- |
| 16x64x128x128 | 64 to 128  | 79.8 ms | 180 ms  | **2.3x** |
| 16x128x64x64  | 128 to 256 | 59.8 ms | 218 ms  | **3.7x** |

### Medium Batched (batch=8, 3x3)

| Input      | Channels  | Flex    | NdArray | Speedup  |
| ---------- | --------- | ------- | ------- | -------- |
| 8x3x64x64  | 3 to 64   | 925 us  | 491 us  | 0.53x    |
| 8x32x64x64 | 32 to 64  | 4.67 ms | 6.44 ms | **1.4x** |
| 8x64x32x32 | 64 to 128 | 3.07 ms | 9.27 ms | **3.0x** |

### Conv1d

| Input      | Kernel | Flex    | NdArray | Speedup  |
| ---------- | ------ | ------- | ------- | -------- |
| 1x16x256   | 3      | 31.4 us | 163 us  | **5.2x** |
| 8x32x512   | 5      | 536 us  | 2.32 ms | **4.3x** |
| 16x64x1024 | 7      | 5.17 ms | 50.7 ms | **9.8x** |

---

## Pooling

### Max Pool 2D

| Input        | Kernel | Flex    | NdArray | Speedup  |
| ------------ | ------ | ------- | ------- | -------- |
| 1x64x56x56   | 3x3 s2 | 135 us  | 165 us  | **1.2x** |
| 8x64x56x56   | 3x3 s2 | 683 us  | 914 us  | **1.3x** |
| 16x128x28x28 | 2x2 s2 | 406 us  | 640 us  | **1.6x** |
| 1x512x14x14  | 2x2 s2 | 90.5 us | 106 us  | **1.2x** |

### Max Pool 2D (ResNet)

| Input         | Kernel | Flex    | NdArray | Speedup  |
| ------------- | ------ | ------- | ------- | -------- |
| 1x64x112x112  | 3x3 s2 | 446 us  | 520 us  | **1.2x** |
| 8x64x112x112  | 3x3 s2 | 2.63 ms | 3.00 ms | **1.1x** |
| 16x64x112x112 | 3x3 s2 | 5.03 ms | 5.91 ms | **1.2x** |

### Avg Pool 2D

| Input        | Kernel | Flex   | NdArray | Speedup  |
| ------------ | ------ | ------ | ------- | -------- |
| 1x64x56x56   | 3x3 s2 | 155 us | 149 us  | ~1x      |
| 8x64x56x56   | 3x3 s2 | 782 us | 889 us  | **1.1x** |
| 16x128x28x28 | 2x2 s2 | 484 us | 480 us  | ~1x      |

### Adaptive Avg Pool 2D

| Input       | Output | Flex    | NdArray | Speedup  |
| ----------- | ------ | ------- | ------- | -------- |
| 1x256x56x56 | 7x7    | 151 us  | 142 us  | 0.94x    |
| 1x512x7x7   | 1x1    | 63.7 us | 68.2 us | **1.1x** |
| 8x512x7x7   | 1x1    | 112 us  | 110 us  | ~1x      |
| 16x2048x7x7 | 1x1    | 286 us  | 289 us  | ~1x      |

### Max Pool 1D

| Input       | Kernel | Flex    | NdArray | Speedup  |
| ----------- | ------ | ------- | ------- | -------- |
| 1x64x256    | 3 s2   | 57.8 us | 79.4 us | **1.4x** |
| 8x128x512   | 3 s2   | 316 us  | 828 us  | **2.6x** |
| 16x256x1024 | 3 s2   | 1.73 ms | 5.41 ms | **3.1x** |

### Kernel Size Comparison (4x64x56x56)

| Kernel | Flex    | NdArray | Speedup  |
| ------ | ------- | ------- | -------- |
| 2x2    | 221 us  | 317 us  | **1.4x** |
| 3x3    | 375 us  | 515 us  | **1.4x** |
| 5x5    | 1.03 ms | 799 us  | 0.78x    |

---

## Transposed Convolutions

### Conv Transpose 2D

| Input          | Output | Flex    | NdArray | Speedup |
| -------------- | ------ | ------- | ------- | ------- |
| 1x64x7x7       | 14x14  | 138 us  | 1.67 ms | **12x** |
| 1x128x14x14    | 28x28  | 533 us  | 12.9 ms | **24x** |
| 1x256x28x28    | 56x56  | 2.49 ms | 209 ms  | **84x** |
| 1x512x7x7 k3s1 | 7x7    | 1.01 ms | 52.6 ms | **52x** |
| 8x64x14x14     | 28x28  | 3.41 ms | 49.7 ms | **15x** |

### DCGAN Generator

| Layer          | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| -------------- | ------- | ------- | -------- | -------- | ----------- |
| 1x1 to 4x4     | 156 us  | 1.43 ms | **9.2x** | 33.1 KB  | 16.4 KB     |
| 4x4 to 8x8     | 234 us  | 3.83 ms | **16x**  | 164 KB   | 32.8 KB     |
| 8x8 to 16x16   | 305 us  | 4.24 ms | **14x**  | 852 KB   | 65.6 KB     |
| 16x16 to 32x32 | 26.4 us | 1.47 ms | **56x**  | 193 KB   | 12.3 KB     |

### Conv Transpose 1D

| Input     | Flex    | NdArray | Speedup |
| --------- | ------- | ------- | ------- |
| 1x64x32   | 17.5 us | 383 us  | **22x** |
| 8x128x64  | 433 us  | 9.02 ms | **21x** |
| 1x256x128 | 337 us  | 8.95 ms | **27x** |

### Conv Transpose 3D

| Input      | Output   | Flex    | NdArray | Speedup |
| ---------- | -------- | ------- | ------- | ------- |
| 1x32x4x4x4 | 8x8x8    | 245 us  | 2.58 ms | **11x** |
| 1x64x8x8x8 | 16x16x16 | 1.41 ms | 47.4 ms | **34x** |

---

## Interpolation

### Nearest

| Input       | Output  | Flex    | NdArray | Speedup  |
| ----------- | ------- | ------- | ------- | -------- |
| 1x3x64x64   | 128x128 | 22.7 us | 138 us  | **6.1x** |
| 1x3x32x32   | 128x128 | 22.8 us | 143 us  | **6.3x** |
| 1x3x256x256 | 128x128 | 22.5 us | 143 us  | **6.3x** |
| 8x3x64x64   | 128x128 | 58.5 us | 307 us  | **5.2x** |
| 1x64x32x32  | 64x64   | 57.9 us | 254 us  | **4.4x** |

### Bilinear

| Input       | Output  | Flex    | NdArray | Speedup  |
| ----------- | ------- | ------- | ------- | -------- |
| 1x3x64x64   | 128x128 | 81.2 us | 161 us  | **2.0x** |
| 1x3x32x32   | 128x128 | 86.5 us | 147 us  | **1.7x** |
| 1x3x256x256 | 128x128 | 83.9 us | 157 us  | **1.9x** |
| 8x3x64x64   | 128x128 | 181 us  | 382 us  | **2.1x** |
| 1x64x32x32  | 64x64   | 108 us  | 309 us  | **2.8x** |

### Bicubic

| Input       | Output  | Flex   | NdArray | Speedup  |
| ----------- | ------- | ------ | ------- | -------- |
| 1x3x64x64   | 128x128 | 159 us | 239 us  | **1.5x** |
| 1x3x32x32   | 128x128 | 160 us | 232 us  | **1.4x** |
| 1x3x256x256 | 128x128 | 159 us | 238 us  | **1.5x** |
| 8x3x64x64   | 128x128 | 894 us | 994 us  | **1.1x** |
| 1x64x32x32  | 64x64   | 616 us | 707 us  | **1.1x** |

---

## Grid Sample 2D

| Input      | Grid  | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---------- | ----- | ------- | ------- | -------- | -------- | ----------- |
| 1x3x32x32  | 32x32 | 16.7 us | 87.6 us | **5.2x** | 12.5 KB  | 12.3 KB     |
| 1x3x64x64  | 64x64 | 66.9 us | 127 us  | **1.9x** | 49.4 KB  | 49.2 KB     |
| 4x3x32x32  | 32x32 | 60.7 us | 152 us  | **2.5x** | 49.4 KB  | 49.2 KB     |
| 1x16x64x64 | 64x64 | 291 us  | 223 us  | 0.77x    | 262 KB   | 262 KB      |

---

## Cross Product & Unfold

### Cross Product

| Shape   | Flex    | NdArray | Speedup  |
| ------- | ------- | ------- | -------- |
| 1Kx3    | 31.1 us | 43.5 us | **1.4x** |
| 64Kx3   | 1.88 ms | 2.78 ms | **1.5x** |
| 256Kx3  | 7.58 ms | 11.1 ms | **1.5x** |
| 64x3x64 | 141 us  | 290 us  | **2.1x** |

### Unfold (1D)

| Input | Window | Step | Flex    | NdArray | Speedup      |
| ----- | ------ | ---- | ------- | ------- | ------------ |
| 1K    | 8      | 1    | 62.2 ns | 120 us  | **~1927x**   |
| 64K   | 8      | 1    | 64.1 ns | 7.55 ms | **~117686x** |
| 64K   | 64     | 1    | 62.2 ns | 8.04 ms | **~129295x** |
| 64K   | 64     | 32   | 62.2 ns | 254 us  | **~4094x**   |

### Unfold (2D/3D)

| Shape    | Dim | Window | Step | Flex    | NdArray | Speedup     |
| -------- | --- | ------ | ---- | ------- | ------- | ----------- |
| 256x256  | 1   | 8      | 1    | 68.7 ns | 871 us  | **~12685x** |
| 256x256  | 1   | 32     | 16   | 62.5 ns | 57.7 us | **~924x**   |
| 1024x256 | 1   | 8      | 1    | 62.8 ns | 3.28 ms | **~52182x** |
| 32x64x64 | 2   | 8      | 4    | 77.1 ns | 424 us  | **~5497x**  |

---

## Deformable Convolutions

### Small/Tiny Inputs

| Input     | Config      | Flex    | NdArray | Speedup  |
| --------- | ----------- | ------- | ------- | -------- |
| 1x3x8x8   | 3 to 8, k3  | 8.72 us | 92.6 us | **11x**  |
| 1x3x8x8   | no mask     | 7.98 us | 78.9 us | **9.9x** |
| 1x3x16x16 | 3 to 16, k3 | 36.4 us | 122 us  | **3.4x** |
| 1x3x16x16 | stride 2    | 10.2 us | 80.8 us | **7.9x** |
| 2x8x16x16 | 8 to 16, k3 | 116 us  | 246 us  | **2.1x** |

### Medium Inputs

| Input      | Config       | Flex   | NdArray | Speedup |
| ---------- | ------------ | ------ | ------- | ------- |
| 1x16x32x32 | 16 to 32, k3 | 826 us | 581 us  | 0.70x   |
| 1x16x32x32 | wg=4         | 840 us | 528 us  | 0.63x   |
| 1x16x32x32 | og=4         | 942 us | 606 us  | 0.64x   |

---

## Attention (Scaled Dot-Product)

Flex auto-selects between two gemm-backed strategies:

- **Naive** (score matrix <= 256K elements): Materializes full [seq_q, seq_kv] score matrix. Two
  large gemm calls per (batch, head) amortize dispatch overhead better than many small tiled calls.
- **Flash** (score matrix > 256K elements): Tiles over KV dimension with online softmax.
  `O(seq_q * TILE_KV)` memory per head instead of `O(seq_q * seq_kv)`.

Both fuse scale + softcap + masking + bias + softmax into a single pass, reducing intermediate
allocations from ~12 (NdArray fallback) to 3.

### Self-Attention

| Config          | Flex    | NdArray | Speedup  |
| --------------- | ------- | ------- | -------- |
| h8, s64, d64    | 180 us  | 534 us  | **3.0x** |
| h12, s128, d64  | 989 us  | 1.61 ms | **1.6x** |
| h12, s256, d64  | 3.79 ms | 6.03 ms | **1.6x** |
| h12, s512, d64  | 14.8 ms | 22.7 ms | **1.5x** |
| h32, s256, d128 | 15.1 ms | 17.5 ms | **1.2x** |
| b4, h12, s128   | 3.96 ms | 5.47 ms | **1.4x** |

### Causal Attention

| Config         | Flex    | NdArray | Speedup  |
| -------------- | ------- | ------- | -------- |
| h12, s128, d64 | 1.01 ms | 1.72 ms | **1.7x** |
| h12, s256, d64 | 3.82 ms | 6.47 ms | **1.7x** |
| h12, s512, d64 | 14.8 ms | 23.8 ms | **1.6x** |

### With Additive Bias (ALiBi-style)

| Config         | Flex    | NdArray | Speedup  |
| -------------- | ------- | ------- | -------- |
| h12, s128, d64 | 1.04 ms | 1.60 ms | **1.5x** |
| h12, s256, d64 | 4.00 ms | 6.15 ms | **1.5x** |

### Cross-Attention (seq_q != seq_k)

| Config            | Flex    | NdArray | Speedup  |
| ----------------- | ------- | ------- | -------- |
| sq128, sk512, d64 | 3.82 ms | 6.14 ms | **1.6x** |
| sq32, sk1024, d64 | 2.05 ms | 3.72 ms | **1.8x** |

---

## Quantized Tensor Operations

All quantized ops (except layout ops) go through a dequantize-op-quantize cycle. Flex stores scales
separately and applies `scale * x_q` directly; NdArray reparses `QuantizedBytes` on every dequantize
call, which dominates the cost.

### Quantize (float to i8)

| Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | -------- | -------- | ----------- |
| 4K   | 6.93 us | 10.4 us | **1.5x** | 20.6 KB  | 24.7 KB     |
| 64K  | 109 us  | 145 us  | **1.3x** | 328 KB   | 393 KB      |
| 1M   | 1.75 ms | 2.31 ms | **1.3x** | 5.24 MB  | 6.29 MB     |

### Dequantize (i8 to float)

| Size | Flex    | NdArray | Speedup   | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | --------- | -------- | ----------- |
| 4K   | 399 ns  | 48.8 us | **~122x** | 16.5 KB  | 24.6 KB     |
| 64K  | 3.73 us | 801 us  | **~215x** | 262 KB   | 393 KB      |
| 1M   | 54.6 us | 13.0 ms | **~238x** | 4.19 MB  | 6.29 MB     |

### q_add (dequant + add + requant)

| Size | Flex    | NdArray | Speedup   | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | --------- | -------- | ----------- |
| 4K   | 1.15 us | 101 us  | **88x**   | 20.6 KB  | 41.0 KB     |
| 64K  | 14.3 us | 1.61 ms | **~113x** | 524 KB   | 655 KB      |
| 1M   | 208 us  | 25.9 ms | **~125x** | 8.39 MB  | 10.5 MB     |

### q_matmul (dequant + matmul + requant)

| Size    | Flex    | NdArray | Speedup | Flex Mem | NdArray Mem |
| ------- | ------- | ------- | ------- | -------- | ----------- |
| 64x64   | 7.10 us | 137 us  | **19x** | 66.5 KB  | 49.3 KB     |
| 256x256 | 147 us  | 1.93 ms | **13x** | 1.05 MB  | 788 KB      |
| 512x512 | 641 us  | 8.07 ms | **13x** | 4.19 MB  | 3.15 MB     |

### q_sum (dequant + sum)

| Size | Flex    | NdArray | Speedup   | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | --------- | -------- | ----------- |
| 4K   | 605 ns  | 50.7 us | **84x**   | 2.13 KB  | 24.6 KB     |
| 64K  | 7.71 us | 834 us  | **~108x** | 262 KB   | 393 KB      |
| 1M   | 194 us  | 13.3 ms | **68x**   | 4.19 MB  | 6.29 MB     |

### q_permute (zero-copy layout op)

| Size      | Flex    | NdArray | Speedup | Flex Mem | NdArray Mem |
| --------- | ------- | ------- | ------- | -------- | ----------- |
| 256x256   | 75.5 ns | 66.1 ns | 0.88x   | 20.5 B   | 4.00 B      |
| 1024x1024 | 77.5 ns | 66.1 ns | 0.85x   | 20.5 B   | 4.00 B      |

### q_argmax (operates on i8 directly)

| Size      | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 67.9 us | 106 us  | **1.6x** | 3.25 KB  | 4.16 KB     |
| 1024x1024 | 137 us  | 1.71 ms | **12x**  | 12.5 KB  | 16.4 KB     |

### q_argmin (operates on i8 directly)

| Size      | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 69.4 us | 106 us  | **1.5x** | 3.25 KB  | 4.16 KB     |
| 1024x1024 | 139 us  | 1.72 ms | **12x**  | 12.5 KB  | 16.4 KB     |

### q_gather (operates on i8 directly for tensor-level quant)

| Size      | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 65.7 us | 155 us  | **2.4x** | 590 KB   | 721 KB      |
| 1024x1024 | 393 us  | 2.40 ms | **6.1x** | 9.44 MB  | 11.5 MB     |

---

## Default Ops (sort, repeat, creation, embedding, predicates)

These ops override burn's default trait implementations with direct storage operations.

### Sort (f32, 1D)

| Size | Flex    | NdArray | Speedup  |
| ---- | ------- | ------- | -------- |
| 4K   | 53.6 us | 123 us  | **2.3x** |
| 64K  | 593 us  | 1.59 ms | **2.7x** |
| 1M   | 8.47 ms | 23.9 ms | **2.8x** |

### Sort (f32, 2D along last dim)

| Size      | Flex    | NdArray | Speedup  |
| --------- | ------- | ------- | -------- |
| 64x64     | 5.72 us | 70.3 us | **12x**  |
| 256x256   | 200 us  | 1.26 ms | **6.3x** |
| 1024x1024 | 1.17 ms | 34.5 ms | **29x**  |

### Argsort (f32, 1D)

| Size | Flex    | NdArray | Speedup  |
| ---- | ------- | ------- | -------- |
| 4K   | 70.2 us | 123 us  | **1.7x** |
| 1M   | 12.8 ms | 26.2 ms | **2.1x** |

### Repeat Dim (f32, 256x256)

| Config            | Flex    | NdArray | Speedup  |
| ----------------- | ------- | ------- | -------- |
| dim0 4x           | 12.7 us | 134 us  | **11x**  |
| dim1 4x           | 11.5 us | 138 us  | **12x**  |
| dim0 8x (512x512) | 98.4 us | 880 us  | **8.9x** |

### Tensor Creation (f32, 1M elements)

| Operation | Flex    | NdArray | Speedup |
| --------- | ------- | ------- | ------- |
| zeros     | 17.7 us | 579 us  | **33x** |
| ones      | 35.7 us | 579 us  | **16x** |
| full      | 35.9 us | 579 us  | **16x** |

### Arange (i64)

| Size | Flex    | NdArray | Speedup |
| ---- | ------- | ------- | ------- |
| 4K   | 1.21 us | 1.21 us | ~1x     |
| 1M   | 299 us  | 280 us  | 0.94x   |

### Embedding (f32)

| Config                  | Flex    | NdArray | Speedup  |
| ----------------------- | ------- | ------- | -------- |
| 30k vocab, d=512, 8x128 | 26.1 us | 150 us  | **5.8x** |
| 50k vocab, d=768, 4x256 | 38.9 us | 188 us  | **4.8x** |

### Predicates (f32, 1M elements)

| Operation | Flex    | NdArray | Speedup  |
| --------- | ------- | ------- | -------- |
| is_nan    | 46.4 us | 73.6 us | **1.6x** |
| is_inf    | 52.6 us | 147 us  | **2.8x** |

---

## FFT (Real FFT)

Compared against `realfft` v3 (backed by `rustfft` v6), the gold-standard pure-Rust FFT library.
NdArray does not implement rfft. `realfft` requires `std`; Flex works in `no_std`.

### 1D rfft

| Size    | Flex (median) | realfft (median) | Ratio |
| ------- | ------------- | ---------------- | ----- |
| n=256   | 841 ns        | 252 ns           | 3.3x  |
| n=1024  | 2.64 us       | 991 ns           | 2.7x  |
| n=4096  | 10.6 us       | 4.37 us          | 2.4x  |
| n=16384 | 45.4 us       | 20.7 us          | 2.2x  |
| n=65536 | 231 us        | 91.2 us          | 2.5x  |

### Batched 2D rfft (along last dim)

| Size      | Flex (median) | realfft (median) | Ratio |
| --------- | ------------- | ---------------- | ----- |
| 16 x 1024 | 59.9 us       | 15.9 us          | 3.8x  |
| 64 x 1024 | 114 us        | 64.0 us          | 1.8x  |
| 256 x 256 | 191 us        | 64.9 us          | 2.9x  |

### 1D irfft (inverse)

| Size    | Flex (median) | realfft (median) | Ratio |
| ------- | ------------- | ---------------- | ----- |
| n=256   | 1.32 us       | 207 ns           | 6.4x  |
| n=1024  | 4.01 us       | 908 ns           | 4.4x  |
| n=4096  | 15.6 us       | 4.24 us          | 3.7x  |
| n=16384 | 63.2 us       | 21.1 us          | 3.0x  |
| n=65536 | 278 us        | 93.0 us          | 3.0x  |

Flex implementation: Cooley-Tukey with mixed radix-4/radix-2, complex packing (forward) / inverse
packing (inverse), compile-time twiddle tables via const fn, SIMD vectorization via macerator,
unrolled small kernels (N=2,4,8), and rayon parallelism across fibers. The remaining gap to rustfft
is due to their hand-tuned per-arch SIMD rewrites, split-radix algorithms, and strength-reduced
modular arithmetic.

---

## Running Benchmarks

```bash
cargo bench --bench attention
cargo bench --bench binary_ops
cargo bench --bench matmul
cargo bench --bench int_ops
cargo bench --bench slice_ops
cargo bench --bench reduce_ops
cargo bench --bench cumulative_ops
cargo bench --bench gather_scatter_ops
cargo bench --bench unary_ops
cargo bench --bench comparison_ops
cargo bench --bench conv_ops
cargo bench --bench pool_ops
cargo bench --bench conv_transpose_ops
cargo bench --bench interpolate_ops
cargo bench --bench cross_unfold_ops
cargo bench --bench deform_conv_ops
cargo bench --bench quantization_ops
cargo bench --bench cat_max_min_ops
cargo bench --bench default_ops
cargo bench --bench fft_ops
```
