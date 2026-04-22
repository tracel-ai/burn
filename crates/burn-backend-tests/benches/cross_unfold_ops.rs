//! Benchmarks for cross and unfold operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench cross_unfold_ops --features simd
//! ```

#[path = "common/mod.rs"]
mod common;
use common::{BencherExt, TestBackend};

use burn_tensor::{Tensor, TensorData, backend::Backend};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Cross and unfold ops Benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

// Cross product requires 3 elements along the specified dimension
fn make_cross_tensor_1d<B: Backend>(batch: usize) -> Tensor<B, 2> {
    let data: Vec<f32> = (0..batch * 3).map(|i| (i % 1000) as f32 / 100.0).collect();
    Tensor::from_data(TensorData::new(data, [batch, 3]), &Default::default())
}

fn make_cross_tensor_2d<B: Backend>(d0: usize, d1: usize) -> Tensor<B, 3> {
    let data: Vec<f32> = (0..d0 * d1 * 3)
        .map(|i| (i % 1000) as f32 / 100.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [d0, 3, d1]), &Default::default())
}

fn make_tensor_1d<B: Backend>(size: usize) -> Tensor<B, 1> {
    let data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_tensor_2d<B: Backend>(rows: usize, cols: usize) -> Tensor<B, 2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn make_tensor_3d<B: Backend>(d0: usize, d1: usize, d2: usize) -> Tensor<B, 3> {
    let size = d0 * d1 * d2;
    let data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
    Tensor::from_data(TensorData::new(data, [d0, d1, d2]), &Default::default())
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "cross")]
            mod cross {
                use super::*;

                #[divan::bench]
                fn _1k_vectors(bencher: Bencher) {
                    let a = make_cross_tensor_1d::<B>(1024);
                    let b = make_cross_tensor_1d::<B>(1024);
                    bencher.bench_synced(|| a.clone().cross(b.clone(), 1));
                }

                #[divan::bench]
                fn _64k_vectors(bencher: Bencher) {
                    let a = make_cross_tensor_1d::<B>(64 * 1024);
                    let b = make_cross_tensor_1d::<B>(64 * 1024);
                    bencher.bench_synced(|| a.clone().cross(b.clone(), 1));
                }

                #[divan::bench]
                fn _256k_vectors(bencher: Bencher) {
                    let a = make_cross_tensor_1d::<B>(256 * 1024);
                    let b = make_cross_tensor_1d::<B>(256 * 1024);
                    bencher.bench_synced(|| a.clone().cross(b.clone(), 1));
                }

                #[divan::bench]
                fn _3d_64x64(bencher: Bencher) {
                    let a = make_cross_tensor_2d::<B>(64, 64);
                    let b = make_cross_tensor_2d::<B>(64, 64);
                    bencher.bench_synced(|| a.clone().cross(b.clone(), 1));
                }
            }

            #[divan::bench_group(name = "unfold")]
            mod unfold {
                use super::*;

                // 1D unfold: sliding window extraction
                #[divan::bench]
                fn _1d_1k_win8_step1(bencher: Bencher) {
                    let t: Tensor<B, 1> = make_tensor_1d::<B>(1024);
                    bencher.bench_synced(|| -> Tensor<B, 2> { t.clone().unfold(0, 8, 1) });
                }

                #[divan::bench]
                fn _1d_64k_win8_step1(bencher: Bencher) {
                    let t: Tensor<B, 1> = make_tensor_1d::<B>(64 * 1024);
                    bencher.bench_synced(|| -> Tensor<B, 2> { t.clone().unfold(0, 8, 1) });
                }

                #[divan::bench]
                fn _1d_64k_win64_step1(bencher: Bencher) {
                    let t: Tensor<B, 1> = make_tensor_1d::<B>(64 * 1024);
                    bencher.bench_synced(|| -> Tensor<B, 2> { t.clone().unfold(0, 64, 1) });
                }

                #[divan::bench]
                fn _1d_64k_win64_step32(bencher: Bencher) {
                    let t: Tensor<B, 1> = make_tensor_1d::<B>(64 * 1024);
                    bencher.bench_synced(|| -> Tensor<B, 2> { t.clone().unfold(0, 64, 32) });
                }

                // 2D unfold along dim 1
                #[divan::bench]
                fn _2d_256x256_dim1_win8_step1(bencher: Bencher) {
                    let t: Tensor<B, 2> = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| -> Tensor<B, 3> { t.clone().unfold(1, 8, 1) });
                }

                #[divan::bench]
                fn _2d_256x256_dim1_win32_step16(bencher: Bencher) {
                    let t: Tensor<B, 2> = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| -> Tensor<B, 3> { t.clone().unfold(1, 32, 16) });
                }

                #[divan::bench]
                fn _2d_1024x256_dim1_win8_step1(bencher: Bencher) {
                    let t: Tensor<B, 2> = make_tensor_2d::<B>(1024, 256);
                    bencher.bench_synced(|| -> Tensor<B, 3> { t.clone().unfold(1, 8, 1) });
                }

                // 3D unfold (batch scenarios)
                #[divan::bench]
                fn _3d_32x64x64_dim2_win8_step4(bencher: Bencher) {
                    let t: Tensor<B, 3> = make_tensor_3d::<B>(32, 64, 64);
                    bencher.bench_synced(|| -> Tensor<B, 4> { t.clone().unfold(2, 8, 4) });
                }
            }
        }
    };
}

bench_backend!(TestBackend, backend, "backend");
