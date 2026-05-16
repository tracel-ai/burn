//! Benchmarks for convolution operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench conv_ops --features simd,gemm
//! ```
//!
//! Memory allocation tracking is enabled via divan's AllocProfiler.

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Tensor, TensorData, module, ops::ConvOptions};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Convolution Benchmarks");
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

fn make_kernel_2d(out_ch: usize, in_ch: usize, kh: usize, kw: usize) -> Tensor<4> {
    let data: Vec<f32> = (0..out_ch * in_ch * kh * kw)
        .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [out_ch, in_ch, kh, kw]),
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

fn make_kernel_1d(out_ch: usize, in_ch: usize, kw: usize) -> Tensor<3> {
    let data: Vec<f32> = (0..out_ch * in_ch * kw)
        .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [out_ch, in_ch, kw]),
        &Default::default(),
    )
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            #[divan::bench_group(name = "conv2d_small")]
            mod conv2d_small {
                use super::*;

                #[divan::bench]
                fn conv2d_1x3x32x32_k3x3(bencher: Bencher) {
                    let x = make_input_2d(1, 3, 32, 32);
                    let w = make_kernel_2d(16, 3, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_1x16x32x32_k3x3(bencher: Bencher) {
                    let x = make_input_2d(1, 16, 32, 32);
                    let w = make_kernel_2d(32, 16, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_1x32x16x16_k3x3(bencher: Bencher) {
                    let x = make_input_2d(1, 32, 16, 16);
                    let w = make_kernel_2d(64, 32, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            #[divan::bench_group(name = "conv2d_medium")]
            mod conv2d_medium {
                use super::*;

                #[divan::bench]
                fn conv2d_8x3x64x64_k3x3(bencher: Bencher) {
                    let x = make_input_2d(8, 3, 64, 64);
                    let w = make_kernel_2d(32, 3, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_8x32x64x64_k3x3(bencher: Bencher) {
                    let x = make_input_2d(8, 32, 64, 64);
                    let w = make_kernel_2d(64, 32, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_8x64x32x32_k3x3(bencher: Bencher) {
                    let x = make_input_2d(8, 64, 32, 32);
                    let w = make_kernel_2d(128, 64, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            #[divan::bench_group(name = "conv2d_large")]
            mod conv2d_large {
                use super::*;

                #[divan::bench]
                fn conv2d_16x64x128x128_k3x3(bencher: Bencher) {
                    let x = make_input_2d(16, 64, 128, 128);
                    let w = make_kernel_2d(128, 64, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_16x128x64x64_k3x3(bencher: Bencher) {
                    let x = make_input_2d(16, 128, 64, 64);
                    let w = make_kernel_2d(256, 128, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            #[divan::bench_group(name = "conv2d_resnet")]
            mod conv2d_resnet {
                use super::*;

                #[divan::bench]
                fn resnet_conv1_1x3x224x224_k7x7_s2(bencher: Bencher) {
                    let x = make_input_2d(1, 3, 224, 224);
                    let w = make_kernel_2d(64, 3, 7, 7);
                    let opts = ConvOptions::new([2, 2], [3, 3], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn resnet_layer1_1x64x56x56_k3x3(bencher: Bencher) {
                    let x = make_input_2d(1, 64, 56, 56);
                    let w = make_kernel_2d(64, 64, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn resnet_layer2_1x128x28x28_k3x3(bencher: Bencher) {
                    let x = make_input_2d(1, 128, 28, 28);
                    let w = make_kernel_2d(128, 128, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn resnet_layer3_1x256x14x14_k3x3(bencher: Bencher) {
                    let x = make_input_2d(1, 256, 14, 14);
                    let w = make_kernel_2d(256, 256, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn resnet_layer4_1x512x7x7_k3x3(bencher: Bencher) {
                    let x = make_input_2d(1, 512, 7, 7);
                    let w = make_kernel_2d(512, 512, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            #[divan::bench_group(name = "conv1d")]
            mod conv1d {
                use super::*;

                #[divan::bench]
                fn conv1d_1x16x256_k3(bencher: Bencher) {
                    let x = make_input_1d(1, 16, 256);
                    let w = make_kernel_1d(32, 16, 3);
                    let opts = ConvOptions::new([1], [1], [1], 1);
                    bencher
                        .bench_synced(|| module::conv1d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv1d_8x32x512_k5(bencher: Bencher) {
                    let x = make_input_1d(8, 32, 512);
                    let w = make_kernel_1d(64, 32, 5);
                    let opts = ConvOptions::new([1], [2], [1], 1);
                    bencher
                        .bench_synced(|| module::conv1d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv1d_16x64x1024_k7(bencher: Bencher) {
                    let x = make_input_1d(16, 64, 1024);
                    let w = make_kernel_1d(128, 64, 7);
                    let opts = ConvOptions::new([1], [3], [1], 1);
                    bencher
                        .bench_synced(|| module::conv1d(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            // Depthwise conv1d (groups == channels_in == channels_out).
            // These flow through the same conv3d_depthwise_impl fast path as
            // conv2d because conv1d expands to a trivial 3D shape (kd=kh=1).
            #[divan::bench_group(name = "conv1d_depthwise")]
            mod conv1d_depthwise {
                use super::*;

                #[divan::bench]
                fn depthwise_k3_8x32x512(bencher: Bencher) {
                    let x = make_input_1d(8, 32, 512);
                    let w = make_kernel_1d(32, 1, 3);
                    let opts = ConvOptions::new([1], [1], [1], 32);
                    bencher
                        .bench_synced(|| module::conv1d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn depthwise_k7_8x64x1024(bencher: Bencher) {
                    let x = make_input_1d(8, 64, 1024);
                    let w = make_kernel_1d(64, 1, 7);
                    let opts = ConvOptions::new([1], [3], [1], 64);
                    bencher
                        .bench_synced(|| module::conv1d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn depthwise_k15_4x128x2048(bencher: Bencher) {
                    // Larger receptive field, dilation 1. Tests the common
                    // ConvNeXt-1d / audio separable conv case.
                    let x = make_input_1d(4, 128, 2048);
                    let w = make_kernel_1d(128, 1, 15);
                    let opts = ConvOptions::new([1], [7], [1], 128);
                    bencher
                        .bench_synced(|| module::conv1d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn depthwise_k3_stride2_8x64x1024(bencher: Bencher) {
                    let x = make_input_1d(8, 64, 1024);
                    let w = make_kernel_1d(64, 1, 3);
                    let opts = ConvOptions::new([2], [1], [1], 64);
                    bencher
                        .bench_synced(|| module::conv1d(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            // Small-channel groups=1 convs (Sobel-style edge filters and
            // first-stage image preprocessors). These were reported as a
            // 2.6-3.3x slowdown vs burn-ndarray before the small-channel
            // fast path landed.
            #[divan::bench_group(name = "conv2d_small_channel")]
            mod conv2d_small_channel {
                use super::*;

                #[divan::bench]
                fn sobel_1x3x488x448_k3x3(bencher: Bencher) {
                    // 3-channel image, same-padding 3x3, 3 output channels.
                    let x = make_input_2d(1, 3, 488, 448);
                    let w = make_kernel_2d(3, 3, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn sobel_1x3x488x448_k1x3(bencher: Bencher) {
                    let x = make_input_2d(1, 3, 488, 448);
                    let w = make_kernel_2d(3, 3, 1, 3);
                    let opts = ConvOptions::new([1, 1], [0, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn sobel_1x3x488x448_k3x1(bencher: Bencher) {
                    let x = make_input_2d(1, 3, 488, 448);
                    let w = make_kernel_2d(3, 3, 3, 1);
                    let opts = ConvOptions::new([1, 1], [1, 0], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn sobel_1x3x488x448_k1x5(bencher: Bencher) {
                    // 1x5 Sobel pass. Separable filter, runs on RGB.
                    let x = make_input_2d(1, 3, 488, 448);
                    let w = make_kernel_2d(3, 3, 1, 5);
                    let opts = ConvOptions::new([1, 1], [0, 2], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn sobel_1x3x488x448_k5x1(bencher: Bencher) {
                    // 5x1 Sobel pass. Separable filter, runs on RGB.
                    let x = make_input_2d(1, 3, 488, 448);
                    let w = make_kernel_2d(3, 3, 5, 1);
                    let opts = ConvOptions::new([1, 1], [2, 0], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn downsample_1x4x128x128_k4x4_s2_to12(bencher: Bencher) {
                    // 4 in, 12 out, 4x4 kernel, stride 2. Small downsample stem.
                    let x = make_input_2d(1, 4, 128, 128);
                    let w = make_kernel_2d(12, 4, 4, 4);
                    let opts = ConvOptions::new([2, 2], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn preproc_1x3x488x448_k5x5_to8(bencher: Bencher) {
                    // First-stage image preprocessor: 3 in, 8 out, 5x5.
                    let x = make_input_2d(1, 3, 488, 448);
                    let w = make_kernel_2d(8, 3, 5, 5);
                    let opts = ConvOptions::new([1, 1], [2, 2], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn mask_1x1x256x256_k3x3_to8(bencher: Bencher) {
                    // Single-channel mask input, early feature extractor.
                    let x = make_input_2d(1, 1, 256, 256);
                    let w = make_kernel_2d(8, 1, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn first_layer_4x3x224x224_k7x7_s2(bencher: Bencher) {
                    // Classic ImageNet first layer: 3 in, 7x7, stride 2.
                    // Output channels = 64 is large so gemm could compete;
                    // this bench shows whether our path is at worst on par.
                    let x = make_input_2d(4, 3, 224, 224);
                    let w = make_kernel_2d(64, 3, 7, 7);
                    let opts = ConvOptions::new([2, 2], [3, 3], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            #[divan::bench_group(name = "conv2d_kernel_sizes")]
            mod conv2d_kernel_sizes {
                use super::*;

                #[divan::bench]
                fn conv2d_k1x1(bencher: Bencher) {
                    let x = make_input_2d(4, 64, 56, 56);
                    let w = make_kernel_2d(128, 64, 1, 1);
                    let opts = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_k3x3(bencher: Bencher) {
                    let x = make_input_2d(4, 64, 56, 56);
                    let w = make_kernel_2d(128, 64, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_k5x5(bencher: Bencher) {
                    let x = make_input_2d(4, 64, 56, 56);
                    let w = make_kernel_2d(128, 64, 5, 5);
                    let opts = ConvOptions::new([1, 1], [2, 2], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_k7x7(bencher: Bencher) {
                    let x = make_input_2d(4, 64, 56, 56);
                    let w = make_kernel_2d(128, 64, 7, 7);
                    let opts = ConvOptions::new([1, 1], [3, 3], [1, 1], 1);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            // Depthwise (groups == channels_in == channels_out). Shapes cover
            // MobileNet/ConvNeXt style depthwise blocks and a ConvNeXt-7x7
            // block reported in a user regression on burn-ndarray parity.
            #[divan::bench_group(name = "conv2d_depthwise")]
            mod conv2d_depthwise {
                use super::*;

                #[divan::bench]
                fn depthwise_3x3_4x32x56x56(bencher: Bencher) {
                    let x = make_input_2d(4, 32, 56, 56);
                    let w = make_kernel_2d(32, 1, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 32);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn depthwise_3x3_4x96x28x28(bencher: Bencher) {
                    let x = make_input_2d(4, 96, 28, 28);
                    let w = make_kernel_2d(96, 1, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 96);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn depthwise_3x3_4x192x14x14(bencher: Bencher) {
                    let x = make_input_2d(4, 192, 14, 14);
                    let w = make_kernel_2d(192, 1, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 192);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn depthwise_7x7_4x24x56x56(bencher: Bencher) {
                    // ConvNeXt-style 7x7 depthwise. This shape was 3x slower
                    // than burn-ndarray before the depthwise fast path landed.
                    let x = make_input_2d(4, 24, 56, 56);
                    let w = make_kernel_2d(24, 1, 7, 7);
                    let opts = ConvOptions::new([1, 1], [3, 3], [1, 1], 24);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn depthwise_7x7_4x48x56x56(bencher: Bencher) {
                    // Another regressed shape (channels_out = 48).
                    let x = make_input_2d(4, 48, 56, 56);
                    let w = make_kernel_2d(48, 1, 7, 7);
                    let opts = ConvOptions::new([1, 1], [3, 3], [1, 1], 48);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn depthwise_3x3_stride2_4x64x56x56(bencher: Bencher) {
                    // Downsampling depthwise 3x3.
                    let x = make_input_2d(4, 64, 56, 56);
                    let w = make_kernel_2d(64, 1, 3, 3);
                    let opts = ConvOptions::new([2, 2], [1, 1], [1, 1], 64);
                    bencher
                        .bench_synced(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
                }
            }
        }
    };
}

bench_backend!(backend, "backend");
