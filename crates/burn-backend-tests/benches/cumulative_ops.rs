//! Benchmarks for cumulative operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench cumulative_ops --features simd
//! ```

#[path = "common/mod.rs"]
mod common;
use common::{BencherExt, TestBackend};

use burn_tensor::{Tensor, TensorData, backend::Backend};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Cumulative ops Benchmarks");
    println!();
    divan::main();
    common::report_failures();
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

            #[divan::bench_group(name = "cumsum")]
            mod cumsum {
                use super::*;

                #[divan::bench]
                fn _1k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024);
                    bencher.bench_synced(|| t.clone().cumsum(0));
                }

                #[divan::bench]
                fn _64k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(64 * 1024);
                    bencher.bench_synced(|| t.clone().cumsum(0));
                }

                #[divan::bench]
                fn _1m(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024 * 1024);
                    bencher.bench_synced(|| t.clone().cumsum(0));
                }

                #[divan::bench]
                fn _256x256_dim0(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().cumsum(0));
                }

                #[divan::bench]
                fn _256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().cumsum(1));
                }

                #[divan::bench]
                fn _1024x1024_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench_synced(|| t.clone().cumsum(1));
                }
            }

            #[divan::bench_group(name = "cumprod")]
            mod cumprod {
                use super::*;

                #[divan::bench]
                fn _1k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024);
                    bencher.bench_synced(|| t.clone().cumprod(0));
                }

                #[divan::bench]
                fn _256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().cumprod(1));
                }
            }

            #[divan::bench_group(name = "cummin")]
            mod cummin {
                use super::*;

                #[divan::bench]
                fn _1k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024);
                    bencher.bench_synced(|| t.clone().cummin(0));
                }

                #[divan::bench]
                fn _256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().cummin(1));
                }

                #[divan::bench]
                fn _1024x1024_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench_synced(|| t.clone().cummin(1));
                }
            }

            #[divan::bench_group(name = "cummax")]
            mod cummax {
                use super::*;

                #[divan::bench]
                fn _1k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024);
                    bencher.bench_synced(|| t.clone().cummax(0));
                }

                #[divan::bench]
                fn _256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().cummax(1));
                }

                #[divan::bench]
                fn _1024x1024_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench_synced(|| t.clone().cummax(1));
                }
            }

            // 3D cumulative (batch scenarios)
            #[divan::bench_group(name = "cumsum_3d")]
            mod cumsum_3d {
                use super::*;

                #[divan::bench]
                fn _batch32_64x64_dim2(bencher: Bencher) {
                    let t = make_tensor_3d::<B>(32, 64, 64);
                    bencher.bench_synced(|| t.clone().cumsum(2));
                }

                #[divan::bench]
                fn _batch32_64x64_dim1(bencher: Bencher) {
                    let t = make_tensor_3d::<B>(32, 64, 64);
                    bencher.bench_synced(|| t.clone().cumsum(1));
                }
            }
        }
    };
}

bench_backend!(TestBackend, backend, "backend");
