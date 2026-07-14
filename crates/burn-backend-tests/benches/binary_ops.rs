//! Benchmarks for binary operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench binary_ops
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Int, Tensor, TensorData};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

// Tensor sizes for benchmarking
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

fn make_int_tensor(size: usize) -> Option<Tensor<1, Int>> {
    common::try_setup(|| {
        let data: Vec<i32> = (0..size).map(|i| (i % 1000) as i32).collect();
        Tensor::from_data(TensorData::new(data, [size]), &Default::default())
    })
}

fn make_int_tensor_2d(rows: usize, cols: usize) -> Option<Tensor<2, Int>> {
    common::try_setup(|| {
        let data: Vec<i32> = (0..rows * cols).map(|i| (i % 1000) as i32).collect();
        Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
    })
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            // Add operations
            #[divan::bench_group(name = "add")]
            mod add {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor(SMALL);
                    let b = make_tensor(SMALL);
                    bencher.bench_synced(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor(MEDIUM);
                    let b = make_tensor(MEDIUM);
                    bencher.bench_synced(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    let b = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone() + b.clone());
                }
            }

            // Mul operations
            #[divan::bench_group(name = "mul")]
            mod mul {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor(SMALL);
                    let b = make_tensor(SMALL);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor(MEDIUM);
                    let b = make_tensor(MEDIUM);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    let b = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }
            }

            // Div operations
            #[divan::bench_group(name = "div")]
            mod div {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    let b = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone() / b.clone());
                }
            }

            // Transposed (non-contiguous)
            #[divan::bench_group(name = "add_transposed")]
            mod add_transposed {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let a = make_tensor_2d(256, 256).transpose();
                    let b = make_tensor_2d(256, 256);
                    bencher.bench_synced(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let a = make_tensor_2d(1024, 1024).transpose();
                    let b = make_tensor_2d(1024, 1024);
                    bencher.bench_synced(|| a.clone() + b.clone());
                }
            }

            // Scalar operations
            #[divan::bench_group(name = "scalar")]
            mod scalar {
                use super::*;

                #[divan::bench]
                fn add_large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone() + 1.5);
                }

                #[divan::bench]
                fn mul_large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone() * 2.0);
                }
            }

            // Powf operations
            #[divan::bench_group(name = "powf")]
            mod powf {
                use super::*;

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor(MEDIUM);
                    let b = make_tensor(MEDIUM);
                    bencher.bench_synced(|| a.clone().powf(b.clone()));
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    let b = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().powf(b.clone()));
                }

                #[divan::bench]
                fn scalar_large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().powf_scalar(2.5));
                }
            }

            // Atan2 operations
            #[divan::bench_group(name = "atan2")]
            mod atan2 {
                use super::*;

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor(MEDIUM);
                    let b = make_tensor(MEDIUM);
                    bencher.bench_synced(|| a.clone().atan2(b.clone()));
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor(LARGE);
                    let b = make_tensor(LARGE);
                    bencher.bench_synced(|| a.clone().atan2(b.clone()));
                }
            }
        }
    };
}

bench_backend!(backend, "backend");

// Int benchmarks
macro_rules! bench_int_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            #[divan::bench_group(name = "int_add")]
            mod int_add {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let Some(a) = make_int_tensor(SMALL) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_int_tensor(SMALL) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let Some(a) = make_int_tensor(MEDIUM) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_int_tensor(MEDIUM) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let Some(a) = make_int_tensor(LARGE) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_int_tensor(LARGE) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() + b.clone());
                }
            }

            #[divan::bench_group(name = "int_mul")]
            mod int_mul {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let Some(a) = make_int_tensor(SMALL) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_int_tensor(SMALL) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let Some(a) = make_int_tensor(MEDIUM) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_int_tensor(MEDIUM) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let Some(a) = make_int_tensor(LARGE) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_int_tensor(LARGE) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() * b.clone());
                }
            }

            #[divan::bench_group(name = "int_div")]
            mod int_div {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let Some(a) = make_int_tensor(LARGE) else {
                        bencher.bench(|| ());
                        return;
                    };
                    // Avoid division by zero
                    let data: Vec<i32> = (0..LARGE).map(|i| (i % 999) as i32 + 1).collect();
                    let b: Tensor<1, Int> =
                        Tensor::from_data(TensorData::new(data, [LARGE]), &Default::default());
                    bencher.bench_synced(|| a.clone() / b.clone());
                }
            }

            #[divan::bench_group(name = "int_add_transposed")]
            mod int_add_transposed {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let Some(a) = make_int_tensor_2d(256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let a = a.transpose();
                    let Some(b) = make_int_tensor_2d(256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let Some(a) = make_int_tensor_2d(1024, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let a = a.transpose();
                    let Some(b) = make_int_tensor_2d(1024, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() + b.clone());
                }
            }

            #[divan::bench_group(name = "int_scalar")]
            mod int_scalar {
                use super::*;

                #[divan::bench]
                fn add_large(bencher: Bencher) {
                    let Some(a) = make_int_tensor(LARGE) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() + 100);
                }

                #[divan::bench]
                fn mul_large(bencher: Bencher) {
                    let Some(a) = make_int_tensor(LARGE) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() * 3);
                }
            }
        }
    };
}

bench_int_backend!(backend_int, "backend_int");
