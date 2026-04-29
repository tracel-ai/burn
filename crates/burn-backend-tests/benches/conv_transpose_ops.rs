//! Benchmarks for transposed convolution operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench conv_transpose_ops --features simd,rayon
//! ```
//!
//! Memory allocation tracking is enabled via divan's AllocProfiler.

#[path = "common/mod.rs"]
mod common;
use common::{BencherExt, TestBackend};

use burn_tensor::ops::ConvTransposeOptions;
use burn_tensor::{Tensor, TensorData, backend::Backend, module};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Conv Transpose Benchmarks");
    println!("Memory allocation tracking enabled");
    println!();
    divan::main();
    common::report_failures();
}

fn make_input_2d<B: Backend>(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Tensor<B, 4> {
    let data: Vec<f32> = (0..batch * channels * height * width)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [batch, channels, height, width]),
        &Default::default(),
    )
}

fn make_weight_2d<B: Backend>(
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> Tensor<B, 4> {
    let data: Vec<f32> = (0..in_channels * out_channels * kernel_h * kernel_w)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [in_channels, out_channels, kernel_h, kernel_w]),
        &Default::default(),
    )
}

fn make_input_1d<B: Backend>(batch: usize, channels: usize, length: usize) -> Tensor<B, 3> {
    let data: Vec<f32> = (0..batch * channels * length)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [batch, channels, length]),
        &Default::default(),
    )
}

fn make_weight_1d<B: Backend>(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
) -> Tensor<B, 3> {
    let data: Vec<f32> = (0..in_channels * out_channels * kernel)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [in_channels, out_channels, kernel]),
        &Default::default(),
    )
}

fn make_input_3d<B: Backend>(
    batch: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
) -> Tensor<B, 5> {
    let data: Vec<f32> = (0..batch * channels * depth * height * width)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [batch, channels, depth, height, width]),
        &Default::default(),
    )
}

fn make_weight_3d<B: Backend>(
    in_channels: usize,
    out_channels: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> Tensor<B, 5> {
    let data: Vec<f32> = (0..in_channels * out_channels * kernel_d * kernel_h * kernel_w)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(
            data,
            [in_channels, out_channels, kernel_d, kernel_h, kernel_w],
        ),
        &Default::default(),
    )
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "conv_transpose2d")]
            mod conv_transpose2d {
                use super::*;

                #[divan::bench]
                fn conv_transpose2d_1x64x7x7_to_14x14(bencher: Bencher) {
                    // Upsample from 7x7 to 14x14 (common in decoder/generator)
                    let x = make_input_2d::<B>(1, 64, 7, 7);
                    let w = make_weight_2d::<B>(64, 64, 4, 4);
                    let opts = ConvTransposeOptions::new([2, 2], [1, 1], [0, 0], [1, 1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose2d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }

                #[divan::bench]
                fn conv_transpose2d_1x128x14x14_to_28x28(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 128, 14, 14);
                    let w = make_weight_2d::<B>(128, 64, 4, 4);
                    let opts = ConvTransposeOptions::new([2, 2], [1, 1], [0, 0], [1, 1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose2d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }

                #[divan::bench]
                fn conv_transpose2d_1x256x28x28_to_56x56(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 256, 28, 28);
                    let w = make_weight_2d::<B>(256, 128, 4, 4);
                    let opts = ConvTransposeOptions::new([2, 2], [1, 1], [0, 0], [1, 1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose2d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }

                #[divan::bench]
                fn conv_transpose2d_8x64x14x14_to_28x28(bencher: Bencher) {
                    // Batch of 8
                    let x = make_input_2d::<B>(8, 64, 14, 14);
                    let w = make_weight_2d::<B>(64, 64, 4, 4);
                    let opts = ConvTransposeOptions::new([2, 2], [1, 1], [0, 0], [1, 1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose2d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }

                #[divan::bench]
                fn conv_transpose2d_1x512x7x7_k3x3_s1(bencher: Bencher) {
                    // No upsampling, just transpose conv
                    let x = make_input_2d::<B>(1, 512, 7, 7);
                    let w = make_weight_2d::<B>(512, 512, 3, 3);
                    let opts = ConvTransposeOptions::new([1, 1], [1, 1], [0, 0], [1, 1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose2d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }
            }

            #[divan::bench_group(name = "conv_transpose2d_gan")]
            mod conv_transpose2d_gan {
                use super::*;

                #[divan::bench]
                fn dcgan_layer1_1x512x1x1_to_4x4(bencher: Bencher) {
                    // DCGAN first layer: project and reshape
                    let x = make_input_2d::<B>(1, 512, 1, 1);
                    let w = make_weight_2d::<B>(512, 256, 4, 4);
                    let opts = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose2d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }

                #[divan::bench]
                fn dcgan_layer2_1x256x4x4_to_8x8(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 256, 4, 4);
                    let w = make_weight_2d::<B>(256, 128, 4, 4);
                    let opts = ConvTransposeOptions::new([2, 2], [1, 1], [0, 0], [1, 1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose2d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }

                #[divan::bench]
                fn dcgan_layer3_1x128x8x8_to_16x16(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 128, 8, 8);
                    let w = make_weight_2d::<B>(128, 64, 4, 4);
                    let opts = ConvTransposeOptions::new([2, 2], [1, 1], [0, 0], [1, 1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose2d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }

                #[divan::bench]
                fn dcgan_layer4_1x64x16x16_to_32x32(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 64, 16, 16);
                    let w = make_weight_2d::<B>(64, 3, 4, 4);
                    let opts = ConvTransposeOptions::new([2, 2], [1, 1], [0, 0], [1, 1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose2d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }
            }

            #[divan::bench_group(name = "conv_transpose1d")]
            mod conv_transpose1d {
                use super::*;

                #[divan::bench]
                fn conv_transpose1d_1x64x32_to_64(bencher: Bencher) {
                    let x = make_input_1d::<B>(1, 64, 32);
                    let w = make_weight_1d::<B>(64, 64, 4);
                    let opts = ConvTransposeOptions::new([2], [1], [0], [1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose1d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }

                #[divan::bench]
                fn conv_transpose1d_8x128x64_to_128(bencher: Bencher) {
                    let x = make_input_1d::<B>(8, 128, 64);
                    let w = make_weight_1d::<B>(128, 64, 4);
                    let opts = ConvTransposeOptions::new([2], [1], [0], [1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose1d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }

                #[divan::bench]
                fn conv_transpose1d_1x256x128_to_256(bencher: Bencher) {
                    let x = make_input_1d::<B>(1, 256, 128);
                    let w = make_weight_1d::<B>(256, 128, 4);
                    let opts = ConvTransposeOptions::new([2], [1], [0], [1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose1d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }
            }

            #[divan::bench_group(name = "conv_transpose3d")]
            mod conv_transpose3d {
                use super::*;

                #[divan::bench]
                fn conv_transpose3d_1x32x4x4x4_to_8x8x8(bencher: Bencher) {
                    let x = make_input_3d::<B>(1, 32, 4, 4, 4);
                    let w = make_weight_3d::<B>(32, 32, 4, 4, 4);
                    let opts =
                        ConvTransposeOptions::new([2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose3d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }

                #[divan::bench]
                fn conv_transpose3d_1x64x8x8x8_to_16x16x16(bencher: Bencher) {
                    let x = make_input_3d::<B>(1, 64, 8, 8, 8);
                    let w = make_weight_3d::<B>(64, 32, 4, 4, 4);
                    let opts =
                        ConvTransposeOptions::new([2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
                    bencher.bench_synced(|| {
                        module::conv_transpose3d::<B>(x.clone(), w.clone(), None, opts.clone())
                    });
                }
            }
        }
    };
}

bench_backend!(TestBackend, backend, "backend");
