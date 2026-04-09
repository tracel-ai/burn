//! Benchmarks comparing Flex vs NdArray backends for quantized tensor operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench quantization_ops
//! ```

use burn_flex::Flex;
use burn_ndarray::NdArray;
use burn_tensor::{Tensor, TensorData, backend::Backend, quantization::QTensorPrimitive};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Comparing Flex vs NdArray backends (quantized ops)");
    println!();
    divan::main();
}

const SMALL: usize = 64 * 64; // 4K elements
const MEDIUM: usize = 256 * 256; // 64K elements
const LARGE: usize = 1024 * 1024; // 1M elements

fn make_tensor<B: Backend>(size: usize) -> Tensor<B, 1> {
    let data: Vec<f32> = (0..size)
        .map(|i| (i % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_matrix<B: Backend>(rows: usize, cols: usize) -> Tensor<B, 2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn make_qtensor<B: Backend>(size: usize) -> Tensor<B, 1> {
    make_tensor::<B>(size)
        .quantize_dynamic(&<B as Backend>::QuantizedTensorPrimitive::default_scheme())
}

fn make_qmatrix<B: Backend>(rows: usize, cols: usize) -> Tensor<B, 2> {
    make_matrix::<B>(rows, cols)
        .quantize_dynamic(&<B as Backend>::QuantizedTensorPrimitive::default_scheme())
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "quantize")]
            mod quantize {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let scheme = <B as Backend>::QuantizedTensorPrimitive::default_scheme();
                    bencher.bench(|| make_tensor::<B>(SMALL).quantize_dynamic(&scheme));
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let scheme = <B as Backend>::QuantizedTensorPrimitive::default_scheme();
                    bencher.bench(|| make_tensor::<B>(MEDIUM).quantize_dynamic(&scheme));
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let scheme = <B as Backend>::QuantizedTensorPrimitive::default_scheme();
                    bencher.bench(|| make_tensor::<B>(LARGE).quantize_dynamic(&scheme));
                }
            }

            #[divan::bench_group(name = "dequantize")]
            mod dequantize {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let qt = make_qtensor::<B>(SMALL);
                    bencher.bench(|| qt.clone().dequantize());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let qt = make_qtensor::<B>(MEDIUM);
                    bencher.bench(|| qt.clone().dequantize());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let qt = make_qtensor::<B>(LARGE);
                    bencher.bench(|| qt.clone().dequantize());
                }
            }

            #[divan::bench_group(name = "q_add")]
            mod q_add {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_qtensor::<B>(SMALL);
                    let b = make_qtensor::<B>(SMALL);
                    bencher.bench(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_qtensor::<B>(MEDIUM);
                    let b = make_qtensor::<B>(MEDIUM);
                    bencher.bench(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_qtensor::<B>(LARGE);
                    let b = make_qtensor::<B>(LARGE);
                    bencher.bench(|| a.clone() + b.clone());
                }
            }

            #[divan::bench_group(name = "q_matmul")]
            mod q_matmul {
                use super::*;

                #[divan::bench]
                fn mat_64x64(bencher: Bencher) {
                    let a = make_qmatrix::<B>(64, 64);
                    let b = make_qmatrix::<B>(64, 64);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn mat_256x256(bencher: Bencher) {
                    let a = make_qmatrix::<B>(256, 256);
                    let b = make_qmatrix::<B>(256, 256);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn mat_512x512(bencher: Bencher) {
                    let a = make_qmatrix::<B>(512, 512);
                    let b = make_qmatrix::<B>(512, 512);
                    bencher.bench(|| a.clone().matmul(b.clone()));
                }
            }

            #[divan::bench_group(name = "q_sum")]
            mod q_sum {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let qt = make_qtensor::<B>(SMALL);
                    bencher.bench(|| qt.clone().sum());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let qt = make_qtensor::<B>(MEDIUM);
                    bencher.bench(|| qt.clone().sum());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let qt = make_qtensor::<B>(LARGE);
                    bencher.bench(|| qt.clone().sum());
                }
            }

            #[divan::bench_group(name = "q_permute")]
            mod q_permute {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let qt = make_qmatrix::<B>(256, 256);
                    bencher.bench(|| qt.clone().permute([1, 0]));
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let qt = make_qmatrix::<B>(1024, 1024);
                    bencher.bench(|| qt.clone().permute([1, 0]));
                }
            }

            #[divan::bench_group(name = "q_argmax")]
            mod q_argmax {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let qt = make_qmatrix::<B>(256, 256);
                    bencher.bench(|| qt.clone().argmax(1));
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let qt = make_qmatrix::<B>(1024, 1024);
                    bencher.bench(|| qt.clone().argmax(1));
                }
            }

            #[divan::bench_group(name = "q_argmin")]
            mod q_argmin {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let qt = make_qmatrix::<B>(256, 256);
                    bencher.bench(|| qt.clone().argmin(1));
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let qt = make_qmatrix::<B>(1024, 1024);
                    bencher.bench(|| qt.clone().argmin(1));
                }
            }

            #[divan::bench_group(name = "q_gather")]
            mod q_gather {
                use super::*;
                use burn_tensor::Tensor;

                fn make_indices<B: Backend>(
                    rows: usize,
                    cols: usize,
                ) -> Tensor<B, 2, burn_tensor::Int> {
                    let data: Vec<i64> = (0..rows * cols).map(|i| (i % cols) as i64).collect();
                    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
                }

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let qt = make_qmatrix::<B>(256, 256);
                    let indices = make_indices::<B>(256, 256);
                    bencher.bench(|| qt.clone().gather(1, indices.clone()));
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let qt = make_qmatrix::<B>(1024, 1024);
                    let indices = make_indices::<B>(1024, 1024);
                    bencher.bench(|| qt.clone().gather(1, indices.clone()));
                }
            }
        }
    };
}

bench_backend!(Flex, flex, "Flex");
bench_backend!(NdArray, ndarray, "NdArray");
