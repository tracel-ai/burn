//! Benchmarks comparing Flex vs NdArray backends for matrix multiplication.
//!
//! Run with:
//! ```bash
//! cargo bench --bench matmul --features simd,gemm
//! ```
//!
//! Memory allocation tracking is enabled via divan's AllocProfiler.

use burn_flex::Flex;
use burn_ndarray::NdArray;
use burn_tensor::{Tensor, TensorData, backend::Backend};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Matrix Multiplication Benchmarks: Flex vs NdArray");
    println!("Memory allocation tracking enabled");
    println!();
    divan::main();
}

fn make_matrix<B: Backend>(rows: usize, cols: usize) -> Tensor<B, 2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn make_batch_matrix<B: Backend>(batch: usize, rows: usize, cols: usize) -> Tensor<B, 3> {
    let data: Vec<f32> = (0..batch * rows * cols)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [batch, rows, cols]),
        &Default::default(),
    )
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            // Square matrices
            #[divan::bench_group(name = "square")]
            mod square {
                use super::*;

                #[divan::bench]
                fn matmul_64x64(bencher: Bencher) {
                    let a = make_matrix::<B>(64, 64);
                    let b = make_matrix::<B>(64, 64);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn matmul_128x128(bencher: Bencher) {
                    let a = make_matrix::<B>(128, 128);
                    let b = make_matrix::<B>(128, 128);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn matmul_256x256(bencher: Bencher) {
                    let a = make_matrix::<B>(256, 256);
                    let b = make_matrix::<B>(256, 256);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn matmul_512x512(bencher: Bencher) {
                    let a = make_matrix::<B>(512, 512);
                    let b = make_matrix::<B>(512, 512);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn matmul_1024x1024(bencher: Bencher) {
                    let a = make_matrix::<B>(1024, 1024);
                    let b = make_matrix::<B>(1024, 1024);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }
            }

            // Rectangular matrices (common in neural networks)
            #[divan::bench_group(name = "rectangular")]
            mod rectangular {
                use super::*;

                // Linear layer: [batch, in] x [in, out]
                #[divan::bench]
                fn linear_256x512_512x256(bencher: Bencher) {
                    let a = make_matrix::<B>(256, 512);
                    let b = make_matrix::<B>(512, 256);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                // Attention: [seq, hidden] x [hidden, seq]
                #[divan::bench]
                fn attention_512x64_64x512(bencher: Bencher) {
                    let a = make_matrix::<B>(512, 64);
                    let b = make_matrix::<B>(64, 512);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                // Wide matrix (embedding lookup style)
                #[divan::bench]
                fn wide_128x1024_1024x128(bencher: Bencher) {
                    let a = make_matrix::<B>(128, 1024);
                    let b = make_matrix::<B>(1024, 128);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }
            }

            // Batched matmul (common in transformers)
            #[divan::bench_group(name = "batched")]
            mod batched {
                use super::*;

                #[divan::bench]
                fn batch8_64x64(bencher: Bencher) {
                    let a = make_batch_matrix::<B>(8, 64, 64);
                    let b = make_batch_matrix::<B>(8, 64, 64);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn batch16_128x128(bencher: Bencher) {
                    let a = make_batch_matrix::<B>(16, 128, 128);
                    let b = make_batch_matrix::<B>(16, 128, 128);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn batch32_64x64(bencher: Bencher) {
                    let a = make_batch_matrix::<B>(32, 64, 64);
                    let b = make_batch_matrix::<B>(32, 64, 64);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                // Transformer attention heads
                #[divan::bench]
                fn heads12_seq512_dim64(bencher: Bencher) {
                    let a = make_batch_matrix::<B>(12, 512, 64);
                    let b = make_batch_matrix::<B>(12, 64, 512);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }
            }

            // Transposed inputs (tests contiguous conversion overhead)
            #[divan::bench_group(name = "transposed")]
            mod transposed {
                use super::*;

                #[divan::bench]
                fn lhs_transposed_256x256(bencher: Bencher) {
                    let a = make_matrix::<B>(256, 256).transpose();
                    let b = make_matrix::<B>(256, 256);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn rhs_transposed_256x256(bencher: Bencher) {
                    let a = make_matrix::<B>(256, 256);
                    let b = make_matrix::<B>(256, 256).transpose();
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn both_transposed_256x256(bencher: Bencher) {
                    let a = make_matrix::<B>(256, 256).transpose();
                    let b = make_matrix::<B>(256, 256).transpose();
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }
            }
        }
    };
}

bench_backend!(Flex, flex, "Flex");
bench_backend!(NdArray, ndarray, "NdArray");

// Integer matmul benchmarks
fn make_int_matrix<B: Backend>(rows: usize, cols: usize) -> Tensor<B, 2, burn_tensor::Int> {
    let data: Vec<i32> = (0..rows * cols).map(|i| (i % 1000) as i32 - 500).collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn make_int_batch_matrix<B: Backend>(
    batch: usize,
    rows: usize,
    cols: usize,
) -> Tensor<B, 3, burn_tensor::Int> {
    let data: Vec<i32> = (0..batch * rows * cols)
        .map(|i| (i % 1000) as i32 - 500)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [batch, rows, cols]),
        &Default::default(),
    )
}

macro_rules! bench_int_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "int_square")]
            mod square {
                use super::*;

                #[divan::bench]
                fn matmul_64x64(bencher: Bencher) {
                    let a = make_int_matrix::<B>(64, 64);
                    let b = make_int_matrix::<B>(64, 64);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn matmul_128x128(bencher: Bencher) {
                    let a = make_int_matrix::<B>(128, 128);
                    let b = make_int_matrix::<B>(128, 128);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn matmul_256x256(bencher: Bencher) {
                    let a = make_int_matrix::<B>(256, 256);
                    let b = make_int_matrix::<B>(256, 256);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn matmul_512x512(bencher: Bencher) {
                    let a = make_int_matrix::<B>(512, 512);
                    let b = make_int_matrix::<B>(512, 512);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }
            }

            #[divan::bench_group(name = "int_batched")]
            mod batched {
                use super::*;

                #[divan::bench]
                fn batch8_64x64(bencher: Bencher) {
                    let a = make_int_batch_matrix::<B>(8, 64, 64);
                    let b = make_int_batch_matrix::<B>(8, 64, 64);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn batch16_128x128(bencher: Bencher) {
                    let a = make_int_batch_matrix::<B>(16, 128, 128);
                    let b = make_int_batch_matrix::<B>(16, 128, 128);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }
            }
        }
    };
}

bench_int_backend!(Flex, flex_int, "Flex_Int");
bench_int_backend!(NdArray, ndarray_int, "NdArray_Int");

// Broadcast matmul benchmarks
fn make_batch_matrix_4d<B: Backend>(
    b1: usize,
    b2: usize,
    rows: usize,
    cols: usize,
) -> Tensor<B, 4> {
    let data: Vec<f32> = (0..b1 * b2 * rows * cols)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [b1, b2, rows, cols]),
        &Default::default(),
    )
}

macro_rules! bench_broadcast_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            // Broadcast: [1, M, K] x [batch, K, N] -> [batch, M, N]
            #[divan::bench]
            fn broadcast_1_vs_8_64x64(bencher: Bencher) {
                let a = make_batch_matrix::<B>(1, 64, 64);
                let b = make_batch_matrix::<B>(8, 64, 64);
                bencher.bench(|| a.clone().matmul(b.clone()));
            }

            // Broadcast: [batch, M, K] x [1, K, N] -> [batch, M, N]
            #[divan::bench]
            fn broadcast_8_vs_1_64x64(bencher: Bencher) {
                let a = make_batch_matrix::<B>(8, 64, 64);
                let b = make_batch_matrix::<B>(1, 64, 64);
                bencher.bench(|| a.clone().matmul(b.clone()));
            }

            // 4D broadcast: [2, 1, M, K] x [1, 4, K, N] -> [2, 4, M, N]
            #[divan::bench]
            fn broadcast_4d_2x1_vs_1x4_32x32(bencher: Bencher) {
                let a = make_batch_matrix_4d::<B>(2, 1, 32, 32);
                let b = make_batch_matrix_4d::<B>(1, 4, 32, 32);
                bencher.bench(|| a.clone().matmul(b.clone()));
            }

            // Larger 4D broadcast
            #[divan::bench]
            fn broadcast_4d_4x1_vs_1x4_64x64(bencher: Bencher) {
                let a = make_batch_matrix_4d::<B>(4, 1, 64, 64);
                let b = make_batch_matrix_4d::<B>(1, 4, 64, 64);
                bencher.bench(|| a.clone().matmul(b.clone()));
            }
        }
    };
}

bench_broadcast_backend!(Flex, flex_broadcast, "Flex_Broadcast");
bench_broadcast_backend!(NdArray, ndarray_broadcast, "NdArray_Broadcast");
