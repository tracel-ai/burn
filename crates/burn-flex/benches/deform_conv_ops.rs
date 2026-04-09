//! Benchmarks comparing Flex vs NdArray backends for deformable convolution.
//!
//! Run with:
//! ```bash
//! cargo bench --bench deform_conv_ops --features simd,rayon,gemm
//! ```
//!
//! Memory allocation tracking is enabled via divan's AllocProfiler.

use burn_flex::Flex;
use burn_ndarray::NdArray;
use burn_tensor::{Tensor, TensorData, backend::Backend, module, ops::DeformConvOptions};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Deformable Convolution Benchmarks: Flex vs NdArray");
    println!("Memory allocation tracking enabled");
    println!();
    divan::main();
}

/// Create input tensor [batch, channels, height, width]
fn make_input<B: Backend>(
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

/// Create weight tensor [out_channels, in_channels/groups, kernel_h, kernel_w]
fn make_weight<B: Backend>(
    out_ch: usize,
    in_ch_per_group: usize,
    kh: usize,
    kw: usize,
) -> Tensor<B, 4> {
    let data: Vec<f32> = (0..out_ch * in_ch_per_group * kh * kw)
        .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [out_ch, in_ch_per_group, kh, kw]),
        &Default::default(),
    )
}

/// Create offset tensor [batch, offset_groups * kernel_h * kernel_w * 2, out_h, out_w]
fn make_offset<B: Backend>(
    batch: usize,
    offset_groups: usize,
    kh: usize,
    kw: usize,
    out_h: usize,
    out_w: usize,
) -> Tensor<B, 4> {
    let channels = offset_groups * kh * kw * 2;
    let data: Vec<f32> = (0..batch * channels * out_h * out_w)
        .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [batch, channels, out_h, out_w]),
        &Default::default(),
    )
}

/// Create mask tensor [batch, offset_groups * kernel_h * kernel_w, out_h, out_w]
fn make_mask<B: Backend>(
    batch: usize,
    offset_groups: usize,
    kh: usize,
    kw: usize,
    out_h: usize,
    out_w: usize,
) -> Tensor<B, 4> {
    let channels = offset_groups * kh * kw;
    let data: Vec<f32> = (0..batch * channels * out_h * out_w)
        .map(|i| (i % 100) as f32 / 100.0)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [batch, channels, out_h, out_w]),
        &Default::default(),
    )
}

/// Create bias tensor [out_channels]
fn make_bias<B: Backend>(out_ch: usize) -> Tensor<B, 1> {
    let data: Vec<f32> = (0..out_ch).map(|i| (i % 10) as f32 / 10.0).collect();
    Tensor::from_data(TensorData::new(data, [out_ch]), &Default::default())
}

/// Compute output dimensions for convolution
fn compute_output_size(
    input_size: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "deform_conv2d_tiny")]
            mod deform_conv2d_tiny {
                use super::*;

                // Tiny: 1x3x8x8 -> 8 output channels
                #[divan::bench(sample_count = 20, min_time = 0)]
                fn deform_conv2d_1x3x8x8_k3x3_c8(bencher: Bencher) {
                    let (batch, in_ch, h, w) = (1, 3, 8, 8);
                    let (out_ch, kh, kw) = (8, 3, 3);
                    let (stride, padding, dilation) = ([1, 1], [1, 1], [1, 1]);
                    let (weight_groups, offset_groups) = (1, 1);

                    let out_h = compute_output_size(h, kh, padding[0], stride[0], dilation[0]);
                    let out_w = compute_output_size(w, kw, padding[1], stride[1], dilation[1]);

                    let x = make_input::<B>(batch, in_ch, h, w);
                    let weight = make_weight::<B>(out_ch, in_ch / weight_groups, kh, kw);
                    let offset = make_offset::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let mask = make_mask::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let bias = make_bias::<B>(out_ch);
                    let opts = DeformConvOptions::new(
                        stride,
                        padding,
                        dilation,
                        weight_groups,
                        offset_groups,
                    );

                    bencher.bench(|| {
                        module::deform_conv2d::<B>(
                            x.clone(),
                            offset.clone(),
                            weight.clone(),
                            Some(mask.clone()),
                            Some(bias.clone()),
                            opts.clone(),
                        )
                    });
                }

                // Tiny without mask
                #[divan::bench(sample_count = 20, min_time = 0)]
                fn deform_conv2d_1x3x8x8_k3x3_no_mask(bencher: Bencher) {
                    let (batch, in_ch, h, w) = (1, 3, 8, 8);
                    let (out_ch, kh, kw) = (8, 3, 3);
                    let (stride, padding, dilation) = ([1, 1], [1, 1], [1, 1]);
                    let (weight_groups, offset_groups) = (1, 1);

                    let out_h = compute_output_size(h, kh, padding[0], stride[0], dilation[0]);
                    let out_w = compute_output_size(w, kw, padding[1], stride[1], dilation[1]);

                    let x = make_input::<B>(batch, in_ch, h, w);
                    let weight = make_weight::<B>(out_ch, in_ch / weight_groups, kh, kw);
                    let offset = make_offset::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let opts = DeformConvOptions::new(
                        stride,
                        padding,
                        dilation,
                        weight_groups,
                        offset_groups,
                    );

                    bencher.bench(|| {
                        module::deform_conv2d::<B>(
                            x.clone(),
                            offset.clone(),
                            weight.clone(),
                            None,
                            None,
                            opts.clone(),
                        )
                    });
                }
            }

            #[divan::bench_group(name = "deform_conv2d_small")]
            mod deform_conv2d_small {
                use super::*;

                // Small: 1x3x16x16 -> 16 output channels
                #[divan::bench(sample_count = 20, min_time = 0)]
                fn deform_conv2d_1x3x16x16_k3x3_c16(bencher: Bencher) {
                    let (batch, in_ch, h, w) = (1, 3, 16, 16);
                    let (out_ch, kh, kw) = (16, 3, 3);
                    let (stride, padding, dilation) = ([1, 1], [1, 1], [1, 1]);
                    let (weight_groups, offset_groups) = (1, 1);

                    let out_h = compute_output_size(h, kh, padding[0], stride[0], dilation[0]);
                    let out_w = compute_output_size(w, kw, padding[1], stride[1], dilation[1]);

                    let x = make_input::<B>(batch, in_ch, h, w);
                    let weight = make_weight::<B>(out_ch, in_ch / weight_groups, kh, kw);
                    let offset = make_offset::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let mask = make_mask::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let bias = make_bias::<B>(out_ch);
                    let opts = DeformConvOptions::new(
                        stride,
                        padding,
                        dilation,
                        weight_groups,
                        offset_groups,
                    );

                    bencher.bench(|| {
                        module::deform_conv2d::<B>(
                            x.clone(),
                            offset.clone(),
                            weight.clone(),
                            Some(mask.clone()),
                            Some(bias.clone()),
                            opts.clone(),
                        )
                    });
                }

                // Small with stride 2
                #[divan::bench(sample_count = 20, min_time = 0)]
                fn deform_conv2d_1x3x16x16_k3x3_stride2(bencher: Bencher) {
                    let (batch, in_ch, h, w) = (1, 3, 16, 16);
                    let (out_ch, kh, kw) = (16, 3, 3);
                    let (stride, padding, dilation) = ([2, 2], [1, 1], [1, 1]);
                    let (weight_groups, offset_groups) = (1, 1);

                    let out_h = compute_output_size(h, kh, padding[0], stride[0], dilation[0]);
                    let out_w = compute_output_size(w, kw, padding[1], stride[1], dilation[1]);

                    let x = make_input::<B>(batch, in_ch, h, w);
                    let weight = make_weight::<B>(out_ch, in_ch / weight_groups, kh, kw);
                    let offset = make_offset::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let mask = make_mask::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let bias = make_bias::<B>(out_ch);
                    let opts = DeformConvOptions::new(
                        stride,
                        padding,
                        dilation,
                        weight_groups,
                        offset_groups,
                    );

                    bencher.bench(|| {
                        module::deform_conv2d::<B>(
                            x.clone(),
                            offset.clone(),
                            weight.clone(),
                            Some(mask.clone()),
                            Some(bias.clone()),
                            opts.clone(),
                        )
                    });
                }

                // Small batch of 2
                #[divan::bench(sample_count = 20, min_time = 0)]
                fn deform_conv2d_2x8x16x16_k3x3_c16(bencher: Bencher) {
                    let (batch, in_ch, h, w) = (2, 8, 16, 16);
                    let (out_ch, kh, kw) = (16, 3, 3);
                    let (stride, padding, dilation) = ([1, 1], [1, 1], [1, 1]);
                    let (weight_groups, offset_groups) = (1, 1);

                    let out_h = compute_output_size(h, kh, padding[0], stride[0], dilation[0]);
                    let out_w = compute_output_size(w, kw, padding[1], stride[1], dilation[1]);

                    let x = make_input::<B>(batch, in_ch, h, w);
                    let weight = make_weight::<B>(out_ch, in_ch / weight_groups, kh, kw);
                    let offset = make_offset::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let mask = make_mask::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let bias = make_bias::<B>(out_ch);
                    let opts = DeformConvOptions::new(
                        stride,
                        padding,
                        dilation,
                        weight_groups,
                        offset_groups,
                    );

                    bencher.bench(|| {
                        module::deform_conv2d::<B>(
                            x.clone(),
                            offset.clone(),
                            weight.clone(),
                            Some(mask.clone()),
                            Some(bias.clone()),
                            opts.clone(),
                        )
                    });
                }
            }

            #[divan::bench_group(name = "deform_conv2d_medium")]
            mod deform_conv2d_medium {
                use super::*;

                // Medium: 1x16x32x32 -> 32 output channels
                #[divan::bench(sample_count = 20, min_time = 0)]
                fn deform_conv2d_1x16x32x32_k3x3_c32(bencher: Bencher) {
                    let (batch, in_ch, h, w) = (1, 16, 32, 32);
                    let (out_ch, kh, kw) = (32, 3, 3);
                    let (stride, padding, dilation) = ([1, 1], [1, 1], [1, 1]);
                    let (weight_groups, offset_groups) = (1, 1);

                    let out_h = compute_output_size(h, kh, padding[0], stride[0], dilation[0]);
                    let out_w = compute_output_size(w, kw, padding[1], stride[1], dilation[1]);

                    let x = make_input::<B>(batch, in_ch, h, w);
                    let weight = make_weight::<B>(out_ch, in_ch / weight_groups, kh, kw);
                    let offset = make_offset::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let mask = make_mask::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let bias = make_bias::<B>(out_ch);
                    let opts = DeformConvOptions::new(
                        stride,
                        padding,
                        dilation,
                        weight_groups,
                        offset_groups,
                    );

                    bencher.bench(|| {
                        module::deform_conv2d::<B>(
                            x.clone(),
                            offset.clone(),
                            weight.clone(),
                            Some(mask.clone()),
                            Some(bias.clone()),
                            opts.clone(),
                        )
                    });
                }

                // Medium with weight groups
                #[divan::bench(sample_count = 20, min_time = 0)]
                fn deform_conv2d_1x16x32x32_k3x3_wg4(bencher: Bencher) {
                    let (batch, in_ch, h, w) = (1, 16, 32, 32);
                    let (out_ch, kh, kw) = (32, 3, 3);
                    let (stride, padding, dilation) = ([1, 1], [1, 1], [1, 1]);
                    let (weight_groups, offset_groups) = (4, 1);

                    let out_h = compute_output_size(h, kh, padding[0], stride[0], dilation[0]);
                    let out_w = compute_output_size(w, kw, padding[1], stride[1], dilation[1]);

                    let x = make_input::<B>(batch, in_ch, h, w);
                    let weight = make_weight::<B>(out_ch, in_ch / weight_groups, kh, kw);
                    let offset = make_offset::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let mask = make_mask::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let bias = make_bias::<B>(out_ch);
                    let opts = DeformConvOptions::new(
                        stride,
                        padding,
                        dilation,
                        weight_groups,
                        offset_groups,
                    );

                    bencher.bench(|| {
                        module::deform_conv2d::<B>(
                            x.clone(),
                            offset.clone(),
                            weight.clone(),
                            Some(mask.clone()),
                            Some(bias.clone()),
                            opts.clone(),
                        )
                    });
                }

                // Medium with offset groups
                #[divan::bench(sample_count = 20, min_time = 0)]
                fn deform_conv2d_1x16x32x32_k3x3_og4(bencher: Bencher) {
                    let (batch, in_ch, h, w) = (1, 16, 32, 32);
                    let (out_ch, kh, kw) = (32, 3, 3);
                    let (stride, padding, dilation) = ([1, 1], [1, 1], [1, 1]);
                    let (weight_groups, offset_groups) = (1, 4);

                    let out_h = compute_output_size(h, kh, padding[0], stride[0], dilation[0]);
                    let out_w = compute_output_size(w, kw, padding[1], stride[1], dilation[1]);

                    let x = make_input::<B>(batch, in_ch, h, w);
                    let weight = make_weight::<B>(out_ch, in_ch / weight_groups, kh, kw);
                    let offset = make_offset::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let mask = make_mask::<B>(batch, offset_groups, kh, kw, out_h, out_w);
                    let bias = make_bias::<B>(out_ch);
                    let opts = DeformConvOptions::new(
                        stride,
                        padding,
                        dilation,
                        weight_groups,
                        offset_groups,
                    );

                    bencher.bench(|| {
                        module::deform_conv2d::<B>(
                            x.clone(),
                            offset.clone(),
                            weight.clone(),
                            Some(mask.clone()),
                            Some(bias.clone()),
                            opts.clone(),
                        )
                    });
                }
            }
        }
    };
}

bench_backend!(Flex, flex, "Flex");
bench_backend!(NdArray, ndarray, "NdArray");
