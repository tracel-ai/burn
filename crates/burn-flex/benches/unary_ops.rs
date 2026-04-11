//! Benchmarks comparing Flex vs NdArray backends for unary operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench unary_ops --features simd
//! ```

use burn_flex::Flex;
use burn_ndarray::NdArray;
use burn_tensor::{Tensor, TensorData, backend::Backend};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Comparing Flex vs NdArray backends for unary ops");
    println!();
    divan::main();
}

const SMALL: usize = 64 * 64; // 4K elements
const MEDIUM: usize = 256 * 256; // 64K elements
const LARGE: usize = 1024 * 1024; // 1M elements

fn make_tensor<B: Backend>(size: usize) -> Tensor<B, 1> {
    // Use values in range [0.1, 1.1] to avoid domain issues for log/sqrt
    let data: Vec<f32> = (0..size)
        .map(|i| 0.1 + (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_tensor_2d<B: Backend>(rows: usize, cols: usize) -> Tensor<B, 2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| 0.1 + (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "exp")]
            mod exp {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor::<B>(SMALL);
                    bencher.bench(|| a.clone().exp());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone().exp());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone().exp());
                }
            }

            #[divan::bench_group(name = "log")]
            mod log {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor::<B>(SMALL);
                    bencher.bench(|| a.clone().log());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone().log());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone().log());
                }
            }

            #[divan::bench_group(name = "sqrt")]
            mod sqrt {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor::<B>(SMALL);
                    bencher.bench(|| a.clone().sqrt());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone().sqrt());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone().sqrt());
                }
            }

            #[divan::bench_group(name = "sin")]
            mod sin {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor::<B>(SMALL);
                    bencher.bench(|| a.clone().sin());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone().sin());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone().sin());
                }
            }

            #[divan::bench_group(name = "cos")]
            mod cos {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor::<B>(SMALL);
                    bencher.bench(|| a.clone().cos());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone().cos());
                }
            }

            #[divan::bench_group(name = "tanh")]
            mod tanh {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor::<B>(SMALL);
                    bencher.bench(|| a.clone().tanh());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone().tanh());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone().tanh());
                }
            }

            #[divan::bench_group(name = "abs")]
            mod abs {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone().abs());
                }
            }

            #[divan::bench_group(name = "recip")]
            mod recip {
                use super::*;

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor::<B>(LARGE);
                    bencher.bench(|| a.clone().recip());
                }
            }

            // Non-contiguous (transposed)
            #[divan::bench_group(name = "exp_transposed")]
            mod exp_transposed {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let a = make_tensor_2d::<B>(256, 256).transpose();
                    bencher.bench(|| a.clone().exp());
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let a = make_tensor_2d::<B>(1024, 1024).transpose();
                    bencher.bench(|| a.clone().exp());
                }
            }
        }
    };
}

bench_backend!(Flex, flex, "Flex");
bench_backend!(NdArray, ndarray, "NdArray");
