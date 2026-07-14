//! Benchmarks for cumulative operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench cumulative_ops --features simd
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Tensor, TensorData};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Cumulative ops Benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

fn make_tensor_1d(size: usize) -> Tensor<1> {
    let data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_tensor_2d(rows: usize, cols: usize) -> Tensor<2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn make_tensor_3d(d0: usize, d1: usize, d2: usize) -> Tensor<3> {
    let size = d0 * d1 * d2;
    let data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
    Tensor::from_data(TensorData::new(data, [d0, d1, d2]), &Default::default())
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            #[divan::bench_group(name = "cumsum")]
            mod cumsum {
                use super::*;

                #[divan::bench]
                fn s_1k(bencher: Bencher) {
                    let t = make_tensor_1d(1024);
                    bencher.bench_synced(|| t.clone().cumsum(0));
                }

                #[divan::bench]
                fn s_64k(bencher: Bencher) {
                    let t = make_tensor_1d(64 * 1024);
                    bencher.bench_synced(|| t.clone().cumsum(0));
                }

                #[divan::bench]
                fn s_1m(bencher: Bencher) {
                    let t = make_tensor_1d(1024 * 1024);
                    bencher.bench_synced(|| t.clone().cumsum(0));
                }

                #[divan::bench]
                fn s_256x256_dim0(bencher: Bencher) {
                    let t = make_tensor_2d(256, 256);
                    bencher.bench_synced(|| t.clone().cumsum(0));
                }

                #[divan::bench]
                fn s_256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_2d(256, 256);
                    bencher.bench_synced(|| t.clone().cumsum(1));
                }

                #[divan::bench]
                fn s_1024x1024_dim1(bencher: Bencher) {
                    let t = make_tensor_2d(1024, 1024);
                    bencher.bench_synced(|| t.clone().cumsum(1));
                }
            }

            #[divan::bench_group(name = "cumprod")]
            mod cumprod {
                use super::*;

                #[divan::bench]
                fn s_1k(bencher: Bencher) {
                    let t = make_tensor_1d(1024);
                    bencher.bench_synced(|| t.clone().cumprod(0));
                }

                #[divan::bench]
                fn s_256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_2d(256, 256);
                    bencher.bench_synced(|| t.clone().cumprod(1));
                }
            }

            #[divan::bench_group(name = "cummin")]
            mod cummin {
                use super::*;

                #[divan::bench]
                fn s_1k(bencher: Bencher) {
                    let t = make_tensor_1d(1024);
                    bencher.bench_synced(|| t.clone().cummin(0));
                }

                #[divan::bench]
                fn s_256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_2d(256, 256);
                    bencher.bench_synced(|| t.clone().cummin(1));
                }

                #[divan::bench]
                fn s_1024x1024_dim1(bencher: Bencher) {
                    let t = make_tensor_2d(1024, 1024);
                    bencher.bench_synced(|| t.clone().cummin(1));
                }
            }

            #[divan::bench_group(name = "cummax")]
            mod cummax {
                use super::*;

                #[divan::bench]
                fn s_1k(bencher: Bencher) {
                    let t = make_tensor_1d(1024);
                    bencher.bench_synced(|| t.clone().cummax(0));
                }

                #[divan::bench]
                fn s_256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_2d(256, 256);
                    bencher.bench_synced(|| t.clone().cummax(1));
                }

                #[divan::bench]
                fn s_1024x1024_dim1(bencher: Bencher) {
                    let t = make_tensor_2d(1024, 1024);
                    bencher.bench_synced(|| t.clone().cummax(1));
                }
            }

            // 3D cumulative (batch scenarios)
            #[divan::bench_group(name = "cumsum_3d")]
            mod cumsum_3d {
                use super::*;

                #[divan::bench]
                fn s_batch32_64x64_dim2(bencher: Bencher) {
                    let t = make_tensor_3d(32, 64, 64);
                    bencher.bench_synced(|| t.clone().cumsum(2));
                }

                #[divan::bench]
                fn s_batch32_64x64_dim1(bencher: Bencher) {
                    let t = make_tensor_3d(32, 64, 64);
                    bencher.bench_synced(|| t.clone().cumsum(1));
                }
            }
        }
    };
}

bench_backend!(backend, "backend");
