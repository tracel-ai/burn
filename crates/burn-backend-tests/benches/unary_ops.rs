//! Benchmarks for unary operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench unary_ops --features simd
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Tensor, TensorData};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Unary ops Benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

const SMALL: usize = 64 * 64; // 4K elements
const MEDIUM: usize = 256 * 256; // 64K elements
const LARGE: usize = 1024 * 1024; // 1M elements

fn make_tensor(size: usize) -> Tensor<1> {
    // Use values in range [0.1, 1.1] to avoid domain issues for log/sqrt
    let data: Vec<f32> = (0..size)
        .map(|i| 0.1 + (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_tensor_2d(rows: usize, cols: usize) -> Tensor<2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| 0.1 + (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            #[divan::bench_group(name = "exp")]
            mod exp {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor(SMALL);
                    bencher.bench_synced(|| a.clone().exp());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor(MEDIUM);
                    bencher.bench_synced(|| a.clone().exp());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().exp());
                }
            }

            #[divan::bench_group(name = "log")]
            mod log {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor(SMALL);
                    bencher.bench_synced(|| a.clone().log());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor(MEDIUM);
                    bencher.bench_synced(|| a.clone().log());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().log());
                }
            }

            #[divan::bench_group(name = "sqrt")]
            mod sqrt {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor(SMALL);
                    bencher.bench_synced(|| a.clone().sqrt());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor(MEDIUM);
                    bencher.bench_synced(|| a.clone().sqrt());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().sqrt());
                }
            }

            #[divan::bench_group(name = "sin")]
            mod sin {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor(SMALL);
                    bencher.bench_synced(|| a.clone().sin());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor(MEDIUM);
                    bencher.bench_synced(|| a.clone().sin());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().sin());
                }
            }

            #[divan::bench_group(name = "cos")]
            mod cos {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor(SMALL);
                    bencher.bench_synced(|| a.clone().cos());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().cos());
                }
            }

            #[divan::bench_group(name = "tanh")]
            mod tanh {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor(SMALL);
                    bencher.bench_synced(|| a.clone().tanh());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor(MEDIUM);
                    bencher.bench_synced(|| a.clone().tanh());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().tanh());
                }
            }

            #[divan::bench_group(name = "abs")]
            mod abs {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().abs());
                }
            }

            #[divan::bench_group(name = "recip")]
            mod recip {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().recip());
                }
            }

            // Non-contiguous (transposed)
            #[divan::bench_group(name = "exp_transposed")]
            mod exp_transposed {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let a = make_tensor_2d(256, 256).transpose();
                    bencher.bench_synced(|| a.clone().exp());
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let a = make_tensor_2d(1024, 1024).transpose();
                    bencher.bench_synced(|| a.clone().exp());
                }
            }
        }
    };
}

bench_backend!(backend, "backend");
