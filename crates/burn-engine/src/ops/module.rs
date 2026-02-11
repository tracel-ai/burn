use burn_backend::{
    ops::{DeformConv2dBackward, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps},
    tensor::{FloatTensor, IntTensor},
};

use crate::Engine;
use crate::backends::*;
use crate::module_op;

impl ModuleOps<Self> for Engine {
    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvOptions<2>,
    ) -> FloatTensor<Self> {
        // TODO: clean up macro that currently always destructures a tuple, and returns one
        module_op!(
            float(x, weight), opt(bias) => (out) {
                (B::conv2d(x, weight, bias, options),)
            }
        )
        .0
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        module_op!(
            float(x, offset, weight), opt(mask, bias) => (out) {
                (B::deform_conv2d(x, offset, weight, mask, bias, options),)
            }
        )
        .0
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let (x_grad, offset_grad, weight_grad, mask_grad, bias_grad) = module_op!(
            float(x, offset, weight, output_grad), opt(mask, bias)
            => (x_grad, offset_grad, weight_grad) opt(mask_grad, bias_grad) {
                let res = B::deform_conv2d_backward(x, offset, weight, mask, bias, output_grad, options);
                (res.x_grad, res.offset_grad, res.weight_grad, res.mask_grad, res.bias_grad)
            }
        );
        DeformConv2dBackward::new(x_grad, offset_grad, weight_grad, mask_grad, bias_grad)
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvOptions<3>,
    ) -> FloatTensor<Self> {
        module_op!(
            float(x, weight), opt(bias) => (out) {
                (B::conv3d(x, weight, bias, options),)
            }
        )
        .0
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        module_op!(
            float(x, weight), opt(bias) => (out) {
                (B::conv_transpose2d(x, weight, bias, options),)
            }
        )
        .0
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        module_op!(
            float(x, weight), opt(bias) => (out) {
                (B::conv_transpose3d(x, weight, bias, options),)
            }
        )
        .0
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        module_op!(
            float(x), opt() => (out) {
                (B::avg_pool2d(x, kernel_size, stride, padding, count_include_pad, ceil_mode),)
            }
        )
        .0
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        module_op!(
            float(x, grad), opt() => (out) {
                (B::avg_pool2d_backward(x, grad, kernel_size, stride, padding, count_include_pad, ceil_mode),)
            }
        )
        .0
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        module_op!(
            float(x), opt() => (out) {
                (B::adaptive_avg_pool2d(x, output_size),)
            }
        )
        .0
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        module_op!(
            float(x, grad), opt() => (out) {
                (B::adaptive_avg_pool2d_backward(x, grad),)
            }
        )
        .0
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        module_op!(
            float(x), opt() => (out) {
                (B::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode),)
            }
        )
        .0
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let (out, indices) = module_op!(
            float(x), opt() => (out, indices) {
                let res = B::max_pool2d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode);
                (res.output, res.indices)
            }
        );
        MaxPool2dWithIndices::new(out, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        let x_grad = module_op!(
            float(x, output_grad), int(indices), opt() => (x_grad) opt() {
                let res = B::max_pool2d_with_indices_backward(x, kernel_size, stride, padding, dilation, ceil_mode, output_grad, indices);
                (res.x_grad,)
            }
        ).0;
        MaxPool2dBackward::new(x_grad)
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: burn_backend::ops::InterpolateOptions,
    ) -> FloatTensor<Self> {
        module_op!(
            float(x), opt() => (out) {
                (B::interpolate(x, output_size, options),)
            }
        )
        .0
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: burn_backend::ops::InterpolateOptions,
    ) -> FloatTensor<Self> {
        module_op!(
            float(x, grad), opt() => (out) {
                (B::interpolate_backward(x, grad, output_size, options),)
            }
        )
        .0
    }
}
