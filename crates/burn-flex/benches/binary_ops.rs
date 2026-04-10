//! Benchmarks comparing Flex vs NdArray backends for binary operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench binary_ops
//! ```

use burn_flex::Flex;
use burn_ndarray::NdArray;
use burn_tensor::{Int, Tensor, TensorData, backend::Backend};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Comparing Flex vs NdArray backends");
    println!();
    divan::main();
}

// Tensor sizes for benchmarking
const SMALL: usize = 64 * 64; // 4K elements
const MEDIUM: usize = 256 * 256; // 64K elements
const LARGE: usize = 1024 * 1024; // 1M elements

fn make_tensor<B: Backend>(size: usize) -> Tensor<B, 1> {
    let data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_tensor_2d<B: Backend>(rows: usize, cols: usize) -> Tensor<B, 2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn make_int_tensor<B: Backend>(size: usize) -> Tensor<B, 1, Int> {
    let data: Vec<i64> = (0..size).map(|i| (i % 1000) as i64).collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_int_tensor_2d<B: Backend>(rows: usize, cols: usize) -> Tensor<B, 2, Int> {
    let data: Vec<i64> = (0..rows * cols).map(|i| (i % 1000) as i64).collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            // Add operations
            #[divan::bench_group(name = "add")]
            mod add {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor::<B>(SMALL);
                    let b = make_tensor::<B>(SMALL);
                    bencher.bench(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor::<B>(MEDIUM);
                    let b = make_tensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    let b = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone() + b.clone());
                }
            }

            // Mul operations
            #[divan::bench_group(name = "mul")]
            mod mul {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor::<B>(SMALL);
                    let b = make_tensor::<B>(SMALL);
                    bencher.bench(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor::<B>(MEDIUM);
                    let b = make_tensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    let b = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone() * b.clone());
                }
            }

            // Div operations
            #[divan::bench_group(name = "div")]
            mod div {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    let b = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone() / b.clone());
                }
            }

            // Transposed (non-contiguous)
            #[divan::bench_group(name = "add_transposed")]
            mod add_transposed {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let a = make_tensor_2d::<B>(256, 256).transpose();
                    let b = make_tensor_2d::<B>(256, 256);
                    bencher.bench(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let a = make_tensor_2d::<B>(1024, 1024).transpose();
                    let b = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench(|| a.clone() + b.clone());
                }
            }

            // Scalar operations
            #[divan::bench_group(name = "scalar")]
            mod scalar {
                use super::*;

                #[divan::bench]
                fn add_large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone() + 1.5);
                }

                #[divan::bench]
                fn mul_large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone() * 2.0);
                }
            }

            // Powf operations
            #[divan::bench_group(name = "powf")]
            mod powf {
                use super::*;

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor::<B>(MEDIUM);
                    let b = make_tensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone().powf(b.clone()));
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    let b = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone().powf(b.clone()));
                }

                #[divan::bench]
                fn scalar_large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone().powf_scalar(2.5));
                }
            }

            // Atan2 operations
            #[divan::bench_group(name = "atan2")]
            mod atan2 {
                use super::*;

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor::<B>(MEDIUM);
                    let b = make_tensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone().atan2(b.clone()));
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    let b = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone().atan2(b.clone()));
                }
            }
        }
    };
}

bench_backend!(Flex, flex, "Flex");
bench_backend!(NdArray, ndarray, "NdArray");

// Int benchmarks
macro_rules! bench_int_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "int_add")]
            mod int_add {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_int_tensor::<B>(SMALL);
                    let b = make_int_tensor::<B>(SMALL);
                    bencher.bench(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_int_tensor::<B>(MEDIUM);
                    let b = make_int_tensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_int_tensor::<B>(LARGE);
                    let b = make_int_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone() + b.clone());
                }
            }

            #[divan::bench_group(name = "int_mul")]
            mod int_mul {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_int_tensor::<B>(SMALL);
                    let b = make_int_tensor::<B>(SMALL);
                    bencher.bench(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_int_tensor::<B>(MEDIUM);
                    let b = make_int_tensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_int_tensor::<B>(LARGE);
                    let b = make_int_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone() * b.clone());
                }
            }

            #[divan::bench_group(name = "int_div")]
            mod int_div {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_int_tensor::<B>(LARGE);
                    // Avoid division by zero
                    let data: Vec<i64> = (0..LARGE).map(|i| (i % 999) as i64 + 1).collect();
                    let b: Tensor<B, 1, Int> =
                        Tensor::from_data(TensorData::new(data, [LARGE]), &Default::default());
                    bencher.bench(|| a.clone() / b.clone());
                }
            }

            #[divan::bench_group(name = "int_add_transposed")]
            mod int_add_transposed {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let a = make_int_tensor_2d::<B>(256, 256).transpose();
                    let b = make_int_tensor_2d::<B>(256, 256);
                    bencher.bench(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let a = make_int_tensor_2d::<B>(1024, 1024).transpose();
                    let b = make_int_tensor_2d::<B>(1024, 1024);
                    bencher.bench(|| a.clone() + b.clone());
                }
            }

            #[divan::bench_group(name = "int_scalar")]
            mod int_scalar {
                use super::*;

                #[divan::bench]
                fn add_large(bencher: Bencher) {
                    let a = make_int_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone() + 100);
                }

                #[divan::bench]
                fn mul_large(bencher: Bencher) {
                    let a = make_int_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone() * 3);
                }
            }
        }
    };
}

bench_int_backend!(Flex, flex_int, "Flex_Int");
bench_int_backend!(NdArray, ndarray_int, "NdArray_Int");
