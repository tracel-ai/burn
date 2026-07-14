//! Benchmarks for comparison and broadcasting operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench comparison_ops
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Tensor, TensorData};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Benchmarking comparison and broadcast operations");
    println!();
    divan::main();
    common::report_failures();
}

const SMALL: usize = 64 * 64; // 4K elements
const MEDIUM: usize = 256 * 256; // 64K elements
const LARGE: usize = 1024 * 1024; // 1M elements

fn make_tensor(size: usize) -> Tensor<1> {
    let data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_tensor_2d(rows: usize, cols: usize) -> Tensor<2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            // Greater comparison
            #[divan::bench_group(name = "greater")]
            mod greater {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor(SMALL);
                    let b = make_tensor(SMALL);
                    bencher.bench_synced(|| a.clone().greater(b.clone()));
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor(MEDIUM);
                    let b = make_tensor(MEDIUM);
                    bencher.bench_synced(|| a.clone().greater(b.clone()));
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    let b = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().greater(b.clone()));
                }
            }

            // Greater scalar comparison
            #[divan::bench_group(name = "greater_elem")]
            mod greater_elem {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().greater_elem(0.5));
                }
            }

            // Equal comparison
            #[divan::bench_group(name = "equal")]
            mod equal {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor(SMALL);
                    let b = make_tensor(SMALL);
                    bencher.bench_synced(|| a.clone().equal(b.clone()));
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    let b = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().equal(b.clone()));
                }
            }

            // Lower comparison
            #[divan::bench_group(name = "lower")]
            mod lower {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    let b = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().lower(b.clone()));
                }
            }

            // Non-contiguous (transposed) comparison
            #[divan::bench_group(name = "greater_transposed")]
            mod greater_transposed {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let a = make_tensor_2d(256, 256).transpose();
                    let b = make_tensor_2d(256, 256);
                    bencher.bench_synced(|| a.clone().greater(b.clone()));
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let a = make_tensor_2d(1024, 1024).transpose();
                    let b = make_tensor_2d(1024, 1024);
                    bencher.bench_synced(|| a.clone().greater(b.clone()));
                }
            }

            // Broadcast comparison: [N, 1] > [1, M] -> [N, M]
            #[divan::bench_group(name = "greater_broadcast")]
            mod greater_broadcast {
                use super::*;

                #[divan::bench]
                fn broadcast_256x256(bencher: Bencher) {
                    let a = make_tensor_2d(256, 1);
                    let b = make_tensor_2d(1, 256);
                    bencher.bench_synced(|| a.clone().greater(b.clone()));
                }

                #[divan::bench]
                fn broadcast_1024x1024(bencher: Bencher) {
                    let a = make_tensor_2d(1024, 1);
                    let b = make_tensor_2d(1, 1024);
                    bencher.bench_synced(|| a.clone().greater(b.clone()));
                }
            }

            // Expand operation
            #[divan::bench_group(name = "expand")]
            mod expand {
                use super::*;

                #[divan::bench]
                fn expand_1_to_1m(bencher: Bencher) {
                    let a = make_tensor_2d(1, 1);
                    bencher.bench_synced(|| a.clone().expand([1000, 1000]));
                }

                #[divan::bench]
                fn expand_row_1024x1_to_1024x1024(bencher: Bencher) {
                    let a = make_tensor_2d(1024, 1);
                    bencher.bench_synced(|| a.clone().expand([1024, 1024]));
                }

                #[divan::bench]
                fn expand_col_1x1024_to_1024x1024(bencher: Bencher) {
                    let a = make_tensor_2d(1, 1024);
                    bencher.bench_synced(|| a.clone().expand([1024, 1024]));
                }
            }

            // Bool operations
            #[divan::bench_group(name = "bool_not")]
            mod bool_not {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    let b = make_tensor(LARGE);
                    let mask = a.greater(b);
                    bencher.bench_synced(|| mask.clone().bool_not());
                }
            }

            #[divan::bench_group(name = "bool_and")]
            mod bool_and {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    let b = make_tensor(LARGE);
                    let mask1 = a.clone().greater(b.clone());
                    let mask2 = a.lower(b);
                    bencher.bench_synced(|| mask1.clone().bool_and(mask2.clone()));
                }
            }
        }
    };
}

bench_backend!(backend, "backend");
