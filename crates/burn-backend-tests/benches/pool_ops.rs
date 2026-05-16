//! Benchmarks for pooling operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench pool_ops --features simd,rayon
//! ```
//!
//! Memory allocation tracking is enabled via divan's AllocProfiler.

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Tensor, TensorData, module};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Pooling Benchmarks");
    println!("Memory allocation tracking enabled");
    println!();
    divan::main();
    common::report_failures();
}

fn make_input_2d(batch: usize, channels: usize, height: usize, width: usize) -> Tensor<4> {
    let data: Vec<f32> = (0..batch * channels * height * width)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [batch, channels, height, width]),
        &Default::default(),
    )
}

fn make_input_1d(batch: usize, channels: usize, length: usize) -> Tensor<3> {
    let data: Vec<f32> = (0..batch * channels * length)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [batch, channels, length]),
        &Default::default(),
    )
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            #[divan::bench_group(name = "max_pool2d")]
            mod max_pool2d {
                use super::*;

                #[divan::bench]
                fn max_pool2d_1x64x56x56_k3x3_s2(bencher: Bencher) {
                    let x = make_input_2d(1, 64, 56, 56);
                    bencher.bench_synced(|| {
                        module::max_pool2d(x.clone(), [3, 3], [2, 2], [1, 1], [1, 1], false)
                    });
                }

                #[divan::bench]
                fn max_pool2d_8x64x56x56_k3x3_s2(bencher: Bencher) {
                    let x = make_input_2d(8, 64, 56, 56);
                    bencher.bench_synced(|| {
                        module::max_pool2d(x.clone(), [3, 3], [2, 2], [1, 1], [1, 1], false)
                    });
                }

                #[divan::bench]
                fn max_pool2d_16x128x28x28_k2x2_s2(bencher: Bencher) {
                    let x = make_input_2d(16, 128, 28, 28);
                    bencher.bench_synced(|| {
                        module::max_pool2d(x.clone(), [2, 2], [2, 2], [0, 0], [1, 1], false)
                    });
                }

                #[divan::bench]
                fn max_pool2d_1x512x14x14_k2x2_s2(bencher: Bencher) {
                    let x = make_input_2d(1, 512, 14, 14);
                    bencher.bench_synced(|| {
                        module::max_pool2d(x.clone(), [2, 2], [2, 2], [0, 0], [1, 1], false)
                    });
                }
            }

            #[divan::bench_group(name = "max_pool2d_resnet")]
            mod max_pool2d_resnet {
                use super::*;

                #[divan::bench]
                fn resnet_maxpool_1x64x112x112_k3x3_s2(bencher: Bencher) {
                    // ResNet initial max pool after first conv
                    let x = make_input_2d(1, 64, 112, 112);
                    bencher.bench_synced(|| {
                        module::max_pool2d(x.clone(), [3, 3], [2, 2], [1, 1], [1, 1], false)
                    });
                }

                #[divan::bench]
                fn resnet_maxpool_8x64x112x112_k3x3_s2(bencher: Bencher) {
                    let x = make_input_2d(8, 64, 112, 112);
                    bencher.bench_synced(|| {
                        module::max_pool2d(x.clone(), [3, 3], [2, 2], [1, 1], [1, 1], false)
                    });
                }

                #[divan::bench]
                fn resnet_maxpool_16x64x112x112_k3x3_s2(bencher: Bencher) {
                    let x = make_input_2d(16, 64, 112, 112);
                    bencher.bench_synced(|| {
                        module::max_pool2d(x.clone(), [3, 3], [2, 2], [1, 1], [1, 1], false)
                    });
                }
            }

            #[divan::bench_group(name = "avg_pool2d")]
            mod avg_pool2d {
                use super::*;

                #[divan::bench]
                fn avg_pool2d_1x64x56x56_k3x3_s2(bencher: Bencher) {
                    let x = make_input_2d(1, 64, 56, 56);
                    bencher.bench_synced(|| {
                        module::avg_pool2d(x.clone(), [3, 3], [2, 2], [1, 1], false, false)
                    });
                }

                #[divan::bench]
                fn avg_pool2d_8x64x56x56_k3x3_s2(bencher: Bencher) {
                    let x = make_input_2d(8, 64, 56, 56);
                    bencher.bench_synced(|| {
                        module::avg_pool2d(x.clone(), [3, 3], [2, 2], [1, 1], false, false)
                    });
                }

                #[divan::bench]
                fn avg_pool2d_16x128x28x28_k2x2_s2(bencher: Bencher) {
                    let x = make_input_2d(16, 128, 28, 28);
                    bencher.bench_synced(|| {
                        module::avg_pool2d(x.clone(), [2, 2], [2, 2], [0, 0], false, false)
                    });
                }
            }

            #[divan::bench_group(name = "adaptive_avg_pool2d")]
            mod adaptive_avg_pool2d {
                use super::*;

                #[divan::bench]
                fn adaptive_avg_pool2d_1x512x7x7_to_1x1(bencher: Bencher) {
                    // Global average pooling (common in ResNet final layer)
                    let x = make_input_2d(1, 512, 7, 7);
                    bencher.bench_synced(|| module::adaptive_avg_pool2d(x.clone(), [1, 1]));
                }

                #[divan::bench]
                fn adaptive_avg_pool2d_8x512x7x7_to_1x1(bencher: Bencher) {
                    let x = make_input_2d(8, 512, 7, 7);
                    bencher.bench_synced(|| module::adaptive_avg_pool2d(x.clone(), [1, 1]));
                }

                #[divan::bench]
                fn adaptive_avg_pool2d_16x2048x7x7_to_1x1(bencher: Bencher) {
                    // ResNet-50/101/152 final layer
                    let x = make_input_2d(16, 2048, 7, 7);
                    bencher.bench_synced(|| module::adaptive_avg_pool2d(x.clone(), [1, 1]));
                }

                #[divan::bench]
                fn adaptive_avg_pool2d_1x256x56x56_to_7x7(bencher: Bencher) {
                    // Downsampling to fixed size
                    let x = make_input_2d(1, 256, 56, 56);
                    bencher.bench_synced(|| module::adaptive_avg_pool2d(x.clone(), [7, 7]));
                }
            }

            #[divan::bench_group(name = "max_pool1d")]
            mod max_pool1d {
                use super::*;

                #[divan::bench]
                fn max_pool1d_1x64x256_k3_s2(bencher: Bencher) {
                    let x = make_input_1d(1, 64, 256);
                    bencher.bench_synced(|| module::max_pool1d(x.clone(), 3, 2, 1, 1, false));
                }

                #[divan::bench]
                fn max_pool1d_8x128x512_k3_s2(bencher: Bencher) {
                    let x = make_input_1d(8, 128, 512);
                    bencher.bench_synced(|| module::max_pool1d(x.clone(), 3, 2, 1, 1, false));
                }

                #[divan::bench]
                fn max_pool1d_16x256x1024_k3_s2(bencher: Bencher) {
                    let x = make_input_1d(16, 256, 1024);
                    bencher.bench_synced(|| module::max_pool1d(x.clone(), 3, 2, 1, 1, false));
                }
            }

            #[divan::bench_group(name = "pool_kernel_sizes")]
            mod pool_kernel_sizes {
                use super::*;

                #[divan::bench]
                fn max_pool2d_k2x2(bencher: Bencher) {
                    let x = make_input_2d(4, 64, 56, 56);
                    bencher.bench_synced(|| {
                        module::max_pool2d(x.clone(), [2, 2], [2, 2], [0, 0], [1, 1], false)
                    });
                }

                #[divan::bench]
                fn max_pool2d_k3x3(bencher: Bencher) {
                    let x = make_input_2d(4, 64, 56, 56);
                    bencher.bench_synced(|| {
                        module::max_pool2d(x.clone(), [3, 3], [2, 2], [1, 1], [1, 1], false)
                    });
                }

                #[divan::bench]
                fn max_pool2d_k5x5(bencher: Bencher) {
                    let x = make_input_2d(4, 64, 56, 56);
                    bencher.bench_synced(|| {
                        module::max_pool2d(x.clone(), [5, 5], [2, 2], [2, 2], [1, 1], false)
                    });
                }
            }
        }
    };
}

bench_backend!(backend, "backend");
