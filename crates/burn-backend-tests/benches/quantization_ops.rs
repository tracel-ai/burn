//! Benchmarks for quantized tensor operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench quantization_ops
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Device, Tensor, TensorData};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Quantized ops Benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

const SMALL: usize = 64 * 64; // 4K elements
const MEDIUM: usize = 256 * 256; // 64K elements
const LARGE: usize = 1024 * 1024; // 1M elements

fn make_tensor(size: usize) -> Tensor<1> {
    let data: Vec<f32> = (0..size)
        .map(|i| (i % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_matrix(rows: usize, cols: usize) -> Tensor<2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

// Returns None (and records a setup failure) on backends without quantization support, so the
// caller can fall back to a no-op bench instead of taking down the whole binary.
fn make_qtensor(size: usize) -> Option<Tensor<1>> {
    common::try_setup(|| {
        make_tensor(size).quantize_dynamic(&Device::default().default_quant_scheme())
    })
}

fn make_qmatrix(rows: usize, cols: usize) -> Option<Tensor<2>> {
    common::try_setup(|| {
        make_matrix(rows, cols).quantize_dynamic(&Device::default().default_quant_scheme())
    })
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            #[divan::bench_group(name = "quantize")]
            mod quantize {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let scheme = Device::default().default_quant_scheme();
                    bencher.bench_synced(|| make_tensor(SMALL).quantize_dynamic(&scheme));
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let scheme = Device::default().default_quant_scheme();
                    bencher.bench_synced(|| make_tensor(MEDIUM).quantize_dynamic(&scheme));
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let scheme = Device::default().default_quant_scheme();
                    bencher.bench_synced(|| make_tensor(LARGE).quantize_dynamic(&scheme));
                }
            }

            #[divan::bench_group(name = "dequantize")]
            mod dequantize {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let Some(qt) = make_qtensor(SMALL) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().dequantize());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let Some(qt) = make_qtensor(MEDIUM) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().dequantize());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let Some(qt) = make_qtensor(LARGE) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().dequantize());
                }
            }

            #[divan::bench_group(name = "q_add")]
            mod q_add {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let Some(a) = make_qtensor(SMALL) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_qtensor(SMALL) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let Some(a) = make_qtensor(MEDIUM) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_qtensor(MEDIUM) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let Some(a) = make_qtensor(LARGE) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_qtensor(LARGE) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone() + b.clone());
                }
            }

            #[divan::bench_group(name = "q_matmul")]
            mod q_matmul {
                use super::*;

                #[divan::bench]
                fn mat_64x64(bencher: Bencher) {
                    let Some(a) = make_qmatrix(64, 64) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_qmatrix(64, 64) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn mat_256x256(bencher: Bencher) {
                    let Some(a) = make_qmatrix(256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_qmatrix(256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone().matmul(b.clone()));
                }

                #[divan::bench]
                fn mat_512x512(bencher: Bencher) {
                    let Some(a) = make_qmatrix(512, 512) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(b) = make_qmatrix(512, 512) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| a.clone().matmul(b.clone()));
                }
            }

            #[divan::bench_group(name = "q_sum")]
            mod q_sum {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let Some(qt) = make_qtensor(SMALL) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().sum());
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let Some(qt) = make_qtensor(MEDIUM) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().sum());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let Some(qt) = make_qtensor(LARGE) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().sum());
                }
            }

            #[divan::bench_group(name = "q_permute")]
            mod q_permute {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let Some(qt) = make_qmatrix(256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().permute([1, 0]));
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let Some(qt) = make_qmatrix(1024, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().permute([1, 0]));
                }
            }

            #[divan::bench_group(name = "q_argmax")]
            mod q_argmax {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let Some(qt) = make_qmatrix(256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().argmax(1));
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let Some(qt) = make_qmatrix(1024, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().argmax(1));
                }
            }

            #[divan::bench_group(name = "q_argmin")]
            mod q_argmin {
                use super::*;

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let Some(qt) = make_qmatrix(256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().argmin(1));
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let Some(qt) = make_qmatrix(1024, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| qt.clone().argmin(1));
                }
            }

            #[divan::bench_group(name = "q_gather")]
            mod q_gather {
                use super::*;
                use burn_tensor::Tensor;

                fn make_indices(rows: usize, cols: usize) -> Tensor<2, burn_tensor::Int> {
                    let data: Vec<i32> = (0..rows * cols).map(|i| (i % cols) as i32).collect();
                    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
                }

                #[divan::bench]
                fn medium_256x256(bencher: Bencher) {
                    let Some(qt) = make_qmatrix(256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let indices = make_indices(256, 256);
                    bencher.bench_synced(|| qt.clone().gather(1, indices.clone()));
                }

                #[divan::bench]
                fn large_1024x1024(bencher: Bencher) {
                    let Some(qt) = make_qmatrix(1024, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let indices = make_indices(1024, 1024);
                    bencher.bench_synced(|| qt.clone().gather(1, indices.clone()));
                }
            }
        }
    };
}

bench_backend!(backend, "backend");
