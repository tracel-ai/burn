//! Benchmarks comparing Flex vs NdArray backends for convolution operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench conv_ops --features simd,gemm
//! ```
//!
//! Memory allocation tracking is enabled via divan's AllocProfiler.

use burn_flex::Flex;
use burn_ndarray::NdArray;
use burn_tensor::{Tensor, TensorData, backend::Backend, module, ops::ConvOptions};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Convolution Benchmarks: Flex vs NdArray");
    println!("Memory allocation tracking enabled");
    println!();
    divan::main();
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

fn make_kernel_2d<B: Backend>(out_ch: usize, in_ch: usize, kh: usize, kw: usize) -> Tensor<B, 4> {
    let data: Vec<f32> = (0..out_ch * in_ch * kh * kw)
        .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [out_ch, in_ch, kh, kw]),
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

fn make_kernel_1d<B: Backend>(out_ch: usize, in_ch: usize, kw: usize) -> Tensor<B, 3> {
    let data: Vec<f32> = (0..out_ch * in_ch * kw)
        .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [out_ch, in_ch, kw]),
        &Default::default(),
    )
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "conv2d_small")]
            mod conv2d_small {
                use super::*;

                #[divan::bench]
                fn conv2d_1x3x32x32_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 3, 32, 32);
                    let w = make_kernel_2d::<B>(16, 3, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_1x16x32x32_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 16, 32, 32);
                    let w = make_kernel_2d::<B>(32, 16, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_1x32x16x16_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 32, 16, 16);
                    let w = make_kernel_2d::<B>(64, 32, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            #[divan::bench_group(name = "conv2d_medium")]
            mod conv2d_medium {
                use super::*;

                #[divan::bench]
                fn conv2d_8x3x64x64_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(8, 3, 64, 64);
                    let w = make_kernel_2d::<B>(32, 3, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_8x32x64x64_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(8, 32, 64, 64);
                    let w = make_kernel_2d::<B>(64, 32, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_8x64x32x32_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(8, 64, 32, 32);
                    let w = make_kernel_2d::<B>(128, 64, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            #[divan::bench_group(name = "conv2d_large")]
            mod conv2d_large {
                use super::*;

                #[divan::bench]
                fn conv2d_16x64x128x128_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(16, 64, 128, 128);
                    let w = make_kernel_2d::<B>(128, 64, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_16x128x64x64_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(16, 128, 64, 64);
                    let w = make_kernel_2d::<B>(256, 128, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            #[divan::bench_group(name = "conv2d_resnet")]
            mod conv2d_resnet {
                use super::*;

                #[divan::bench]
                fn resnet_conv1_1x3x224x224_k7x7_s2(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 3, 224, 224);
                    let w = make_kernel_2d::<B>(64, 3, 7, 7);
                    let opts = ConvOptions::new([2, 2], [3, 3], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn resnet_layer1_1x64x56x56_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 64, 56, 56);
                    let w = make_kernel_2d::<B>(64, 64, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn resnet_layer2_1x128x28x28_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 128, 28, 28);
                    let w = make_kernel_2d::<B>(128, 128, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn resnet_layer3_1x256x14x14_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 256, 14, 14);
                    let w = make_kernel_2d::<B>(256, 256, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn resnet_layer4_1x512x7x7_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(1, 512, 7, 7);
                    let w = make_kernel_2d::<B>(512, 512, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            #[divan::bench_group(name = "conv1d")]
            mod conv1d {
                use super::*;

                #[divan::bench]
                fn conv1d_1x16x256_k3(bencher: Bencher) {
                    let x = make_input_1d::<B>(1, 16, 256);
                    let w = make_kernel_1d::<B>(32, 16, 3);
                    let opts = ConvOptions::new([1], [1], [1], 1);
                    bencher.bench(|| module::conv1d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv1d_8x32x512_k5(bencher: Bencher) {
                    let x = make_input_1d::<B>(8, 32, 512);
                    let w = make_kernel_1d::<B>(64, 32, 5);
                    let opts = ConvOptions::new([1], [2], [1], 1);
                    bencher.bench(|| module::conv1d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv1d_16x64x1024_k7(bencher: Bencher) {
                    let x = make_input_1d::<B>(16, 64, 1024);
                    let w = make_kernel_1d::<B>(128, 64, 7);
                    let opts = ConvOptions::new([1], [3], [1], 1);
                    bencher.bench(|| module::conv1d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }
            }

            #[divan::bench_group(name = "conv2d_kernel_sizes")]
            mod conv2d_kernel_sizes {
                use super::*;

                #[divan::bench]
                fn conv2d_k1x1(bencher: Bencher) {
                    let x = make_input_2d::<B>(4, 64, 56, 56);
                    let w = make_kernel_2d::<B>(128, 64, 1, 1);
                    let opts = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_k3x3(bencher: Bencher) {
                    let x = make_input_2d::<B>(4, 64, 56, 56);
                    let w = make_kernel_2d::<B>(128, 64, 3, 3);
                    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_k5x5(bencher: Bencher) {
                    let x = make_input_2d::<B>(4, 64, 56, 56);
                    let w = make_kernel_2d::<B>(128, 64, 5, 5);
                    let opts = ConvOptions::new([1, 1], [2, 2], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }

                #[divan::bench]
                fn conv2d_k7x7(bencher: Bencher) {
                    let x = make_input_2d::<B>(4, 64, 56, 56);
                    let w = make_kernel_2d::<B>(128, 64, 7, 7);
                    let opts = ConvOptions::new([1, 1], [3, 3], [1, 1], 1);
                    bencher.bench(|| module::conv2d::<B>(x.clone(), w.clone(), None, opts.clone()));
                }
            }
        }
    };
}

bench_backend!(Flex, flex, "Flex");
bench_backend!(NdArray, ndarray, "NdArray");
