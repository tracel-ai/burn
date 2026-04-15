//! Module operations for the Flex backend.
//!
//! These operations power neural network modules like convolutions and pooling.

use crate::ops::{conv, conv_transpose, deform_conv, interpolate, pool};
use crate::{Flex, FlexTensor, Layout};
use burn_backend::{
    DType, Element, TensorMetadata,
    ops::{
        AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
        DeformConvOptions, FloatTensorOps, IntTensorOps, InterpolateMode, InterpolateOptions,
        MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
    },
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::{Bytes, Shape};
use bytemuck::Pod;

/// Cast a tensor from half-precision type E to f32.
pub(crate) fn cast_to_f32<E: Element + Pod + Copy>(
    tensor: FlexTensor,
    to_f32: fn(E) -> f32,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let data: &[E] = tensor.storage();
    let f32_data: alloc::vec::Vec<f32> = data.iter().map(|&v| to_f32(v)).collect();
    let bytes = Bytes::from_elems(f32_data);
    FlexTensor::new(bytes, Layout::contiguous(shape), DType::F32)
}

/// Cast a tensor from f32 back to half-precision type E.
pub(crate) fn cast_from_f32<E: Element + Pod + Copy>(
    tensor: FlexTensor,
    from_f32: fn(f32) -> E,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let data: &[f32] = tensor.storage();
    let half_data: alloc::vec::Vec<E> = data.iter().map(|&v| from_f32(v)).collect();
    let bytes = Bytes::from_elems(half_data);
    FlexTensor::new(bytes, Layout::contiguous(shape), E::dtype())
}

impl ModuleOps<Flex> for Flex {
    fn conv1d(
        x: FloatTensor<Flex>,
        weight: FloatTensor<Flex>,
        bias: Option<FloatTensor<Flex>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => conv::conv1d_f32(x, weight, bias, &options),
            DType::F64 => conv::conv1d_f64(x, weight, bias, &options),
            DType::F16 => conv::conv1d_f16(x, weight, bias, &options),
            DType::BF16 => conv::conv1d_bf16(x, weight, bias, &options),
            dtype => panic!("conv1d: unsupported dtype {:?}", dtype),
        }
    }

    fn conv2d(
        x: FloatTensor<Flex>,
        weight: FloatTensor<Flex>,
        bias: Option<FloatTensor<Flex>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => conv::conv2d_f32(x, weight, bias, &options),
            DType::F64 => conv::conv2d_f64(x, weight, bias, &options),
            DType::F16 => conv::conv2d_f16(x, weight, bias, &options),
            DType::BF16 => conv::conv2d_bf16(x, weight, bias, &options),
            dtype => panic!("conv2d: unsupported dtype {:?}", dtype),
        }
    }

    fn deform_conv2d(
        x: FloatTensor<Flex>,
        offset: FloatTensor<Flex>,
        weight: FloatTensor<Flex>,
        mask: Option<FloatTensor<Flex>>,
        bias: Option<FloatTensor<Flex>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => deform_conv::deform_conv2d_f32(
                x,
                offset,
                weight,
                mask,
                bias,
                options.stride,
                options.padding,
                options.dilation,
                options.weight_groups,
                options.offset_groups,
            ),
            DType::F64 => deform_conv::deform_conv2d_f64(
                x,
                offset,
                weight,
                mask,
                bias,
                options.stride,
                options.padding,
                options.dilation,
                options.weight_groups,
                options.offset_groups,
            ),
            DType::F16 => {
                use burn_std::f16;
                let result = deform_conv::deform_conv2d_f32(
                    cast_to_f32(x, f16::to_f32),
                    cast_to_f32(offset, f16::to_f32),
                    cast_to_f32(weight, f16::to_f32),
                    mask.map(|m| cast_to_f32(m, f16::to_f32)),
                    bias.map(|b| cast_to_f32(b, f16::to_f32)),
                    options.stride,
                    options.padding,
                    options.dilation,
                    options.weight_groups,
                    options.offset_groups,
                );
                cast_from_f32(result, f16::from_f32)
            }
            DType::BF16 => {
                use burn_std::bf16;
                let result = deform_conv::deform_conv2d_f32(
                    cast_to_f32(x, bf16::to_f32),
                    cast_to_f32(offset, bf16::to_f32),
                    cast_to_f32(weight, bf16::to_f32),
                    mask.map(|m| cast_to_f32(m, bf16::to_f32)),
                    bias.map(|b| cast_to_f32(b, bf16::to_f32)),
                    options.stride,
                    options.padding,
                    options.dilation,
                    options.weight_groups,
                    options.offset_groups,
                );
                cast_from_f32(result, bf16::from_f32)
            }
            dtype => panic!("deform_conv2d: unsupported dtype {:?}", dtype),
        }
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Flex>,
        offset: FloatTensor<Flex>,
        weight: FloatTensor<Flex>,
        mask: Option<FloatTensor<Flex>>,
        bias: Option<FloatTensor<Flex>>,
        output_grad: FloatTensor<Flex>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Flex> {
        let (x_grad, offset_grad, weight_grad, mask_grad, bias_grad) = match x.dtype() {
            DType::F32 => deform_conv::deform_conv2d_backward_f32(
                x,
                offset,
                weight,
                mask,
                bias,
                output_grad,
                options.stride,
                options.padding,
                options.dilation,
                options.weight_groups,
                options.offset_groups,
            ),
            DType::F16 => {
                use burn_std::f16;
                let (xg, og, wg, mg, bg) = deform_conv::deform_conv2d_backward_f32(
                    cast_to_f32(x, f16::to_f32),
                    cast_to_f32(offset, f16::to_f32),
                    cast_to_f32(weight, f16::to_f32),
                    mask.map(|m| cast_to_f32(m, f16::to_f32)),
                    bias.map(|b| cast_to_f32(b, f16::to_f32)),
                    cast_to_f32(output_grad, f16::to_f32),
                    options.stride,
                    options.padding,
                    options.dilation,
                    options.weight_groups,
                    options.offset_groups,
                );
                (
                    cast_from_f32(xg, f16::from_f32),
                    cast_from_f32(og, f16::from_f32),
                    cast_from_f32(wg, f16::from_f32),
                    mg.map(|m| cast_from_f32(m, f16::from_f32)),
                    bg.map(|b| cast_from_f32(b, f16::from_f32)),
                )
            }
            DType::BF16 => {
                use burn_std::bf16;
                let (xg, og, wg, mg, bg) = deform_conv::deform_conv2d_backward_f32(
                    cast_to_f32(x, bf16::to_f32),
                    cast_to_f32(offset, bf16::to_f32),
                    cast_to_f32(weight, bf16::to_f32),
                    mask.map(|m| cast_to_f32(m, bf16::to_f32)),
                    bias.map(|b| cast_to_f32(b, bf16::to_f32)),
                    cast_to_f32(output_grad, bf16::to_f32),
                    options.stride,
                    options.padding,
                    options.dilation,
                    options.weight_groups,
                    options.offset_groups,
                );
                (
                    cast_from_f32(xg, bf16::from_f32),
                    cast_from_f32(og, bf16::from_f32),
                    cast_from_f32(wg, bf16::from_f32),
                    mg.map(|m| cast_from_f32(m, bf16::from_f32)),
                    bg.map(|b| cast_from_f32(b, bf16::from_f32)),
                )
            }
            // f64 backward computed via f32: precision loss for large/small values.
            // A native f64 implementation would require duplicating ~400 lines of
            // deform_conv2d_backward. f64 deform_conv is rare in practice.
            DType::F64 => {
                let to = |v: f64| v as f32;
                let from = |v: f32| v as f64;
                let (xg, og, wg, mg, bg) = deform_conv::deform_conv2d_backward_f32(
                    cast_to_f32(x, to),
                    cast_to_f32(offset, to),
                    cast_to_f32(weight, to),
                    mask.map(|m| cast_to_f32(m, to)),
                    bias.map(|b| cast_to_f32(b, to)),
                    cast_to_f32(output_grad, to),
                    options.stride,
                    options.padding,
                    options.dilation,
                    options.weight_groups,
                    options.offset_groups,
                );
                (
                    cast_from_f32(xg, from),
                    cast_from_f32(og, from),
                    cast_from_f32(wg, from),
                    mg.map(|m| cast_from_f32(m, from)),
                    bg.map(|b| cast_from_f32(b, from)),
                )
            }
            dtype => panic!("deform_conv2d_backward: unsupported dtype {:?}", dtype),
        };
        DeformConv2dBackward::new(x_grad, offset_grad, weight_grad, mask_grad, bias_grad)
    }

    fn conv3d(
        x: FloatTensor<Flex>,
        weight: FloatTensor<Flex>,
        bias: Option<FloatTensor<Flex>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => conv::conv3d_f32(x, weight, bias, &options),
            DType::F64 => conv::conv3d_f64(x, weight, bias, &options),
            DType::F16 => conv::conv3d_f16(x, weight, bias, &options),
            DType::BF16 => conv::conv3d_bf16(x, weight, bias, &options),
            dtype => panic!("conv3d: unsupported dtype {:?}", dtype),
        }
    }

    fn conv_transpose1d(
        x: FloatTensor<Flex>,
        weight: FloatTensor<Flex>,
        bias: Option<FloatTensor<Flex>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => conv_transpose::conv_transpose1d_f32(x, weight, bias, &options),
            DType::F64 => conv_transpose::conv_transpose1d_f64(x, weight, bias, &options),
            DType::F16 => conv_transpose::conv_transpose1d_f16(x, weight, bias, &options),
            DType::BF16 => conv_transpose::conv_transpose1d_bf16(x, weight, bias, &options),
            dtype => panic!("conv_transpose1d: unsupported dtype {:?}", dtype),
        }
    }

    fn conv_transpose2d(
        x: FloatTensor<Flex>,
        weight: FloatTensor<Flex>,
        bias: Option<FloatTensor<Flex>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => conv_transpose::conv_transpose2d_f32(x, weight, bias, &options),
            DType::F64 => conv_transpose::conv_transpose2d_f64(x, weight, bias, &options),
            DType::F16 => conv_transpose::conv_transpose2d_f16(x, weight, bias, &options),
            DType::BF16 => conv_transpose::conv_transpose2d_bf16(x, weight, bias, &options),
            dtype => panic!("conv_transpose2d: unsupported dtype {:?}", dtype),
        }
    }

    fn conv_transpose3d(
        x: FloatTensor<Flex>,
        weight: FloatTensor<Flex>,
        bias: Option<FloatTensor<Flex>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => conv_transpose::conv_transpose3d_f32(x, weight, bias, &options),
            DType::F64 => conv_transpose::conv_transpose3d_f64(x, weight, bias, &options),
            DType::F16 => conv_transpose::conv_transpose3d_f16(x, weight, bias, &options),
            DType::BF16 => conv_transpose::conv_transpose3d_bf16(x, weight, bias, &options),
            dtype => panic!("conv_transpose3d: unsupported dtype {:?}", dtype),
        }
    }

    fn avg_pool2d(
        x: FloatTensor<Flex>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => pool::avg_pool2d_f32(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            ),
            DType::F64 => pool::avg_pool2d_f64(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            ),
            DType::F16 => pool::avg_pool2d_f16(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            ),
            DType::BF16 => pool::avg_pool2d_bf16(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            ),
            dtype => panic!("avg_pool2d: unsupported dtype {:?}", dtype),
        }
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Flex>,
        grad: FloatTensor<Flex>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        _divisor_override: bool,
    ) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => pool::avg_pool2d_backward_f32(
                x,
                grad,
                kernel_size,
                stride,
                padding,
                count_include_pad,
            ),
            DType::F64 => pool::avg_pool2d_backward_f64(
                x,
                grad,
                kernel_size,
                stride,
                padding,
                count_include_pad,
            ),
            DType::F16 => pool::avg_pool2d_backward_f16(
                x,
                grad,
                kernel_size,
                stride,
                padding,
                count_include_pad,
            ),
            DType::BF16 => pool::avg_pool2d_backward_bf16(
                x,
                grad,
                kernel_size,
                stride,
                padding,
                count_include_pad,
            ),
            dtype => panic!("avg_pool2d_backward: unsupported dtype {:?}", dtype),
        }
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Flex>, output_size: [usize; 2]) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => pool::adaptive_avg_pool2d_f32(x, output_size),
            DType::F64 => pool::adaptive_avg_pool2d_f64(x, output_size),
            DType::F16 => pool::adaptive_avg_pool2d_f16(x, output_size),
            DType::BF16 => pool::adaptive_avg_pool2d_bf16(x, output_size),
            dtype => panic!("adaptive_avg_pool2d: unsupported dtype {:?}", dtype),
        }
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Flex>,
        grad: FloatTensor<Flex>,
    ) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => pool::adaptive_avg_pool2d_backward_f32(x, grad),
            DType::F64 => pool::adaptive_avg_pool2d_backward_f64(x, grad),
            DType::F16 => pool::adaptive_avg_pool2d_backward_f16(x, grad),
            DType::BF16 => pool::adaptive_avg_pool2d_backward_bf16(x, grad),
            dtype => panic!(
                "adaptive_avg_pool2d_backward: unsupported dtype {:?}",
                dtype
            ),
        }
    }

    fn max_pool2d(
        x: FloatTensor<Flex>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Flex> {
        match x.dtype() {
            DType::F32 => {
                pool::max_pool2d_f32(x, kernel_size, stride, padding, dilation, ceil_mode)
            }
            DType::F64 => {
                pool::max_pool2d_f64(x, kernel_size, stride, padding, dilation, ceil_mode)
            }
            DType::F16 => {
                pool::max_pool2d_f16(x, kernel_size, stride, padding, dilation, ceil_mode)
            }
            DType::BF16 => {
                pool::max_pool2d_bf16(x, kernel_size, stride, padding, dilation, ceil_mode)
            }
            dtype => panic!("max_pool2d: unsupported dtype {:?}", dtype),
        }
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Flex>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Flex> {
        let (output, indices) = match x.dtype() {
            DType::F32 => pool::max_pool2d_with_indices_f32(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ),
            DType::F64 => pool::max_pool2d_with_indices_f64(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ),
            DType::F16 => pool::max_pool2d_with_indices_f16(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ),
            DType::BF16 => pool::max_pool2d_with_indices_bf16(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ),
            dtype => panic!("max_pool2d_with_indices: unsupported dtype {:?}", dtype),
        };
        MaxPool2dWithIndices::new(output, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Flex>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
        output_grad: FloatTensor<Flex>,
        indices: IntTensor<Flex>,
    ) -> MaxPool2dBackward<Flex> {
        let x_grad = match x.dtype() {
            DType::F32 => pool::max_pool2d_backward_f32(x, output_grad, indices),
            DType::F64 => pool::max_pool2d_backward_f64(x, output_grad, indices),
            DType::F16 => pool::max_pool2d_backward_f16(x, output_grad, indices),
            DType::BF16 => pool::max_pool2d_backward_bf16(x, output_grad, indices),
            dtype => panic!(
                "max_pool2d_with_indices_backward: unsupported dtype {:?}",
                dtype
            ),
        };
        MaxPool2dBackward::new(x_grad)
    }

    fn interpolate(
        x: FloatTensor<Flex>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Flex> {
        match (options.mode, x.dtype()) {
            (InterpolateMode::Nearest, DType::F32) => {
                interpolate::interpolate_nearest_f32(x, output_size, options.align_corners)
            }
            (InterpolateMode::Nearest, DType::F64) => {
                interpolate::interpolate_nearest_f64(x, output_size, options.align_corners)
            }
            (InterpolateMode::Nearest, DType::F16) => {
                interpolate::interpolate_nearest_f16(x, output_size, options.align_corners)
            }
            (InterpolateMode::Nearest, DType::BF16) => {
                interpolate::interpolate_nearest_bf16(x, output_size, options.align_corners)
            }
            (InterpolateMode::Bilinear, DType::F32) => {
                interpolate::interpolate_bilinear_f32(x, output_size, options.align_corners)
            }
            (InterpolateMode::Bilinear, DType::F64) => {
                interpolate::interpolate_bilinear_f64(x, output_size, options.align_corners)
            }
            (InterpolateMode::Bilinear, DType::F16) => {
                interpolate::interpolate_bilinear_f16(x, output_size, options.align_corners)
            }
            (InterpolateMode::Bilinear, DType::BF16) => {
                interpolate::interpolate_bilinear_bf16(x, output_size, options.align_corners)
            }
            (InterpolateMode::Bicubic, DType::F32) => {
                interpolate::interpolate_bicubic_f32(x, output_size, options.align_corners)
            }
            (InterpolateMode::Bicubic, DType::F64) => {
                interpolate::interpolate_bicubic_f64(x, output_size, options.align_corners)
            }
            (InterpolateMode::Bicubic, DType::F16) => {
                interpolate::interpolate_bicubic_f16(x, output_size, options.align_corners)
            }
            (InterpolateMode::Bicubic, DType::BF16) => {
                interpolate::interpolate_bicubic_bf16(x, output_size, options.align_corners)
            }
            (InterpolateMode::Lanczos3, DType::F32) => {
                interpolate::interpolate_lanczos3_f32(x, output_size, options.align_corners)
            }
            (InterpolateMode::Lanczos3, DType::F64) => {
                interpolate::interpolate_lanczos3_f64(x, output_size, options.align_corners)
            }
            (InterpolateMode::Lanczos3, DType::F16) => {
                interpolate::interpolate_lanczos3_f16(x, output_size, options.align_corners)
            }
            (InterpolateMode::Lanczos3, DType::BF16) => {
                interpolate::interpolate_lanczos3_bf16(x, output_size, options.align_corners)
            }
            (mode, dtype) => panic!(
                "interpolate: unsupported mode {:?} / dtype {:?}",
                mode, dtype
            ),
        }
    }

    fn interpolate_backward(
        x: FloatTensor<Flex>,
        grad: FloatTensor<Flex>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Flex> {
        match (options.mode, x.dtype()) {
            (InterpolateMode::Nearest, DType::F32) => {
                interpolate::interpolate_nearest_backward_f32(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (InterpolateMode::Nearest, DType::F64) => {
                interpolate::interpolate_nearest_backward_f64(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (InterpolateMode::Nearest, DType::F16) => {
                interpolate::interpolate_nearest_backward_f16(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (InterpolateMode::Nearest, DType::BF16) => {
                interpolate::interpolate_nearest_backward_bf16(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (InterpolateMode::Bilinear, DType::F32) => {
                interpolate::interpolate_bilinear_backward_f32(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (InterpolateMode::Bilinear, DType::F64) => {
                interpolate::interpolate_bilinear_backward_f64(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (InterpolateMode::Bilinear, DType::F16) => {
                interpolate::interpolate_bilinear_backward_f16(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (InterpolateMode::Bilinear, DType::BF16) => {
                interpolate::interpolate_bilinear_backward_bf16(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (InterpolateMode::Bicubic, DType::F32) => {
                interpolate::interpolate_bicubic_backward_f32(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (InterpolateMode::Bicubic, DType::F64) => {
                interpolate::interpolate_bicubic_backward_f64(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (InterpolateMode::Bicubic, DType::F16) => {
                interpolate::interpolate_bicubic_backward_f16(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (InterpolateMode::Bicubic, DType::BF16) => {
                interpolate::interpolate_bicubic_backward_bf16(
                    x,
                    grad,
                    output_size,
                    options.align_corners,
                )
            }
            (mode, dtype) => {
                panic!(
                    "interpolate_backward: unsupported mode {:?} / dtype {:?}",
                    mode, dtype
                )
            }
        }
    }

    fn attention(
        query: FloatTensor<Flex>,
        key: FloatTensor<Flex>,
        value: FloatTensor<Flex>,
        mask: Option<BoolTensor<Flex>>,
        attn_bias: Option<FloatTensor<Flex>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Flex> {
        crate::ops::attention::attention(query, key, value, mask, attn_bias, options)
    }

    fn rfft(signal: FloatTensor<Flex>, dim: usize) -> (FloatTensor<Flex>, FloatTensor<Flex>) {
        match signal.dtype() {
            DType::F32 => crate::ops::fft::rfft_f32(signal, dim),
            DType::F64 => crate::ops::fft::rfft_f64(signal, dim),
            DType::F16 => crate::ops::fft::rfft_f16(signal, dim),
            DType::BF16 => crate::ops::fft::rfft_bf16(signal, dim),
            dtype => panic!("rfft: unsupported dtype {:?}", dtype),
        }
    }

    fn irfft(
        spectrum_re: FloatTensor<Flex>,
        spectrum_im: FloatTensor<Flex>,
        dim: usize,
    ) -> FloatTensor<Flex> {
        match spectrum_re.dtype() {
            DType::F32 => crate::ops::fft::irfft_f32(spectrum_re, spectrum_im, dim),
            DType::F64 => crate::ops::fft::irfft_f64(spectrum_re, spectrum_im, dim),
            DType::F16 => crate::ops::fft::irfft_f16(spectrum_re, spectrum_im, dim),
            DType::BF16 => crate::ops::fft::irfft_bf16(spectrum_re, spectrum_im, dim),
            dtype => panic!("irfft: unsupported dtype {:?}", dtype),
        }
    }

    fn embedding(weights: FloatTensor<Flex>, indices: IntTensor<Flex>) -> FloatTensor<Flex> {
        let [batch_size, seq_length] = indices.shape().dims();
        let [_, d_model] = weights.shape().dims();

        let indices = Flex::int_reshape(indices, Shape::from(alloc::vec![batch_size * seq_length]));
        let output = Flex::float_select(weights, 0, indices);
        Flex::float_reshape(
            output,
            Shape::from(alloc::vec![batch_size, seq_length, d_model]),
        )
    }

    fn layer_norm(
        tensor: FloatTensor<Flex>,
        gamma: FloatTensor<Flex>,
        beta: Option<FloatTensor<Flex>>,
        epsilon: f64,
    ) -> FloatTensor<Flex> {
        crate::ops::activation::layer_norm(tensor, gamma, beta, epsilon)
    }

    fn embedding_backward(
        weights: FloatTensor<Flex>,
        output_grad: FloatTensor<Flex>,
        indices: IntTensor<Flex>,
    ) -> FloatTensor<Flex> {
        let [batch_size, seq_length] = indices.shape().dims();
        let [n_embeddings, d_model] = weights.shape().dims();
        let dtype = output_grad.dtype();

        let indices = Flex::int_reshape(indices, Shape::from(alloc::vec![batch_size * seq_length]));
        let output_grad = Flex::float_reshape(
            output_grad,
            Shape::from(alloc::vec![batch_size * seq_length, d_model]),
        );
        let grad = Flex::float_zeros(
            Shape::from(alloc::vec![n_embeddings, d_model]),
            &Default::default(),
            dtype.into(),
        );
        Flex::float_select_add(grad, 0, indices, output_grad)
    }
}
