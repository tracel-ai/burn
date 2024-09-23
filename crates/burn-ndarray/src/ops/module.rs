use super::{
    adaptive_avgpool::{adaptive_avg_pool2d, adaptive_avg_pool2d_backward},
    avgpool::{avg_pool2d, avg_pool2d_backward},
    conv::{conv2d, conv3d, conv_transpose2d, conv_transpose3d},
    deform_conv::{backward::deform_conv2d_backward, deform_conv2d},
    interpolate::{bicubic_interpolate, bilinear_interpolate, nearest_interpolate},
    maxpool::{max_pool2d, max_pool2d_backward, max_pool2d_with_indices},
};
use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor, NdArray};
use crate::{element::QuantElement, ops::interpolate::nearest_interpolate_backward};
use burn_tensor::ops::*;

impl<E: FloatNdArrayElement, Q: QuantElement> ModuleOps<Self> for NdArray<E, Q> {
    fn conv2d(
        x: NdArrayTensor<E>,
        weight: NdArrayTensor<E>,
        bias: Option<NdArrayTensor<E>>,
        options: ConvOptions<2>,
    ) -> NdArrayTensor<E> {
        conv2d::<E, Q>(x, weight, bias, options)
    }

    fn deform_conv2d(
        x: NdArrayTensor<E>,
        offset: NdArrayTensor<E>,
        weight: NdArrayTensor<E>,
        mask: Option<NdArrayTensor<E>>,
        bias: Option<NdArrayTensor<E>>,
        options: DeformConvOptions<2>,
    ) -> NdArrayTensor<E> {
        deform_conv2d::<E>(x, offset, weight, mask, bias, options)
    }

    fn deform_conv2d_backward(
        x: NdArrayTensor<E>,
        offset: NdArrayTensor<E>,
        weight: NdArrayTensor<E>,
        mask: Option<NdArrayTensor<E>>,
        bias: Option<NdArrayTensor<E>>,
        output_grad: NdArrayTensor<E>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        deform_conv2d_backward(x, offset, weight, mask, bias, output_grad, options)
    }

    fn conv_transpose2d(
        x: NdArrayTensor<E>,
        weight: NdArrayTensor<E>,
        bias: Option<NdArrayTensor<E>>,
        options: ConvTransposeOptions<2>,
    ) -> NdArrayTensor<E> {
        conv_transpose2d(x, weight, bias, options)
    }

    fn avg_pool2d(
        x: NdArrayTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> NdArrayTensor<E> {
        avg_pool2d(x, kernel_size, stride, padding, count_include_pad)
    }

    fn avg_pool2d_backward(
        x: NdArrayTensor<E>,
        grad: NdArrayTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> NdArrayTensor<E> {
        avg_pool2d_backward(x, grad, kernel_size, stride, padding, count_include_pad)
    }

    fn max_pool2d(
        x: NdArrayTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> NdArrayTensor<E> {
        max_pool2d::<E, Q>(x, kernel_size, stride, padding, dilation)
    }

    fn max_pool2d_with_indices(
        x: NdArrayTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<NdArray<E, Q>> {
        let (output, indices) =
            max_pool2d_with_indices::<E, Q>(x, kernel_size, stride, padding, dilation);

        MaxPool2dWithIndices::new(output, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: NdArrayTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: NdArrayTensor<E>,
        indices: NdArrayTensor<i64>,
    ) -> MaxPool2dBackward<NdArray<E, Q>> {
        MaxPool2dBackward::new(max_pool2d_backward(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            output_grad,
            indices,
        ))
    }

    fn adaptive_avg_pool2d(x: NdArrayTensor<E>, output_size: [usize; 2]) -> NdArrayTensor<E> {
        adaptive_avg_pool2d(x, output_size)
    }

    fn adaptive_avg_pool2d_backward(
        x: NdArrayTensor<E>,
        grad: NdArrayTensor<E>,
    ) -> NdArrayTensor<E> {
        adaptive_avg_pool2d_backward(x, grad)
    }

    fn interpolate(
        x: NdArrayTensor<E>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> NdArrayTensor<E> {
        match options.mode {
            InterpolateMode::Nearest => nearest_interpolate(x, output_size),
            InterpolateMode::Bilinear => bilinear_interpolate(x, output_size),
            InterpolateMode::Bicubic => bicubic_interpolate(x, output_size),
        }
    }

    fn interpolate_backward(
        x: NdArrayTensor<E>,
        grad: NdArrayTensor<E>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> NdArrayTensor<E> {
        match options.mode {
            InterpolateMode::Nearest => nearest_interpolate_backward(x, grad, output_size),
            InterpolateMode::Bilinear => {
                panic!("bilinear interpolation backward is not supported for ndarray backend")
            }
            InterpolateMode::Bicubic => {
                panic!("bicubic interpolation backward is not supported for ndarray backend")
            }
        }
    }

    fn conv3d(
        x: NdArrayTensor<E>,
        weight: NdArrayTensor<E>,
        bias: Option<NdArrayTensor<E>>,
        options: ConvOptions<3>,
    ) -> NdArrayTensor<E> {
        conv3d::<E, Q>(x, weight, bias, options)
    }

    fn conv_transpose3d(
        x: NdArrayTensor<E>,
        weight: NdArrayTensor<E>,
        bias: Option<NdArrayTensor<E>>,
        options: ConvTransposeOptions<3>,
    ) -> NdArrayTensor<E> {
        conv_transpose3d(x, weight, bias, options)
    }
}
