use burn_backend::{
    ops::{
        DeformConv2dBackward, MaxPool1dBackward, MaxPool1dWithIndices, MaxPool2dBackward,
        MaxPool2dWithIndices, ModuleOps,
    },
    tensor::{FloatTensor, IntTensor},
};

use crate::Dispatch;
use crate::backends::*;

impl ModuleOps<Self> for Dispatch {
    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvOptions<2>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float)],
            opt_inputs[bias],
            outputs[out],
            {
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
            inputs[(x, float), (offset, float), (weight, float)],
            opt_inputs[mask, bias],
            outputs[out],
            {
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
            inputs[(x, float), (offset, float), (weight, float), (output_grad, float)],
            opt_inputs[mask, bias],
            outputs[(x_grad, Float), (offset_grad, Float), (weight_grad, Float)],
            opt_outputs[mask_grad, bias_grad],
            {
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
            inputs[(x, float), (weight, float)],
            opt_inputs[bias],
            outputs[out],
            {
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
            inputs[(x, float), (weight, float)],
            opt_inputs[bias],
            outputs[out],
            {
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
            inputs[(x, float), (weight, float)],
            opt_inputs[bias],
            outputs[out],
            {
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
        module_op!(inputs[(x, float)], outputs[out], {
            (B::avg_pool2d(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            ),)
        })
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
            inputs[(x, float), (grad, float)],
            outputs[out],
            {
                (B::avg_pool2d_backward(x, grad, kernel_size, stride, padding, count_include_pad, ceil_mode),)
            }
        )
        .0
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        module_op!(inputs[(x, float)], outputs[out], {
            (B::adaptive_avg_pool2d(x, output_size),)
        })
        .0
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (grad, float)],
            outputs[out],
            {
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
        module_op!(inputs[(x, float)], outputs[out], {
            (B::max_pool2d(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ),)
        })
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
            inputs[(x, float)],
            outputs[(out, Float), (indices, Int)],
            {
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
            inputs[(x, float), (output_grad, float), (indices, int)],
            outputs[out],
            {
                let res = B::max_pool2d_with_indices_backward(x, kernel_size, stride, padding, dilation, ceil_mode, output_grad, indices);
                (res.x_grad,)
            }
        )
        .0;
        MaxPool2dBackward::new(x_grad)
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: burn_backend::ops::InterpolateOptions,
    ) -> FloatTensor<Self> {
        module_op!(inputs[(x, float)], outputs[out], {
            (B::interpolate(x, output_size, options),)
        })
        .0
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: burn_backend::ops::InterpolateOptions,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (grad, float)],
            outputs[out],
            {
                (B::interpolate_backward(x, grad, output_size, options),)
            }
        )
        .0
    }

    fn embedding(weights: FloatTensor<Self>, indices: IntTensor<Self>) -> FloatTensor<Self> {
        module_op!(
            inputs[(weights, float), (indices, int)],
            outputs[out],
            {
                (B::embedding(weights, indices),)
            }
        )
        .0
    }

    fn embedding_backward(
        weights: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(weights, float), (output_grad, float), (indices, int)],
            outputs[out],
            {
                (B::embedding_backward(weights, output_grad, indices),)
            }
        )
        .0
    }

    fn conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvOptions<1>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float)],
            opt_inputs[bias],
            outputs[out],
            {
                (B::conv1d(x, weight, bias, options),)
            }
        )
        .0
    }

    fn conv1d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<1>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv1d_x_backward(x, weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv1d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<1>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv1d_weight_backward(x, weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv1d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (bias, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv1d_bias_backward(x, bias, output_grad),)
            }
        )
        .0
    }

    fn conv2d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<2>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv2d_x_backward(x, weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv2d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<2>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv2d_weight_backward(x, weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv2d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (bias, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv2d_bias_backward(x, bias, output_grad),)
            }
        )
        .0
    }

    fn conv3d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<3>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv3d_x_backward(x, weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv3d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<3>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv3d_weight_backward(x, weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv3d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (bias, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv3d_bias_backward(x, bias, output_grad),)
            }
        )
        .0
    }

    fn conv_transpose1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvTransposeOptions<1>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float)],
            opt_inputs[bias],
            outputs[out],
            {
                (B::conv_transpose1d(x, weight, bias, options),)
            }
        )
        .0
    }

    fn conv_transpose1d_x_backward(
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<1>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv_transpose1d_x_backward(weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv_transpose1d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<1>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv_transpose1d_weight_backward(x, weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv_transpose1d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (bias, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv_transpose1d_bias_backward(x, bias, output_grad),)
            }
        )
        .0
    }

    fn conv_transpose2d_x_backward(
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv_transpose2d_x_backward(weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv_transpose2d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv_transpose2d_weight_backward(x, weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv_transpose2d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (bias, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv_transpose2d_bias_backward(x, bias, output_grad),)
            }
        )
        .0
    }

    fn conv_transpose3d_x_backward(
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv_transpose3d_x_backward(weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv_transpose3d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (weight, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv_transpose3d_weight_backward(x, weight, output_grad, options),)
            }
        )
        .0
    }

    fn conv_transpose3d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (bias, float), (output_grad, float)],
            outputs[out],
            {
                (B::conv_transpose3d_bias_backward(x, bias, output_grad),)
            }
        )
        .0
    }

    fn unfold4d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        options: burn_backend::ops::UnfoldOptions,
    ) -> FloatTensor<Self> {
        module_op!(inputs[(x, float)], outputs[out], {
            (B::unfold4d(x, kernel_size, options),)
        })
        .0
    }

    fn avg_pool1d(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        module_op!(inputs[(x, float)], outputs[out], {
            (B::avg_pool1d(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            ),)
        })
        .0
    }

    fn avg_pool1d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (grad, float)],
            outputs[out],
            {
                (B::avg_pool1d_backward(x, grad, kernel_size, stride, padding, count_include_pad, ceil_mode),)
            }
        )
        .0
    }

    fn adaptive_avg_pool1d(x: FloatTensor<Self>, output_size: usize) -> FloatTensor<Self> {
        module_op!(inputs[(x, float)], outputs[out], {
            (B::adaptive_avg_pool1d(x, output_size),)
        })
        .0
    }

    fn adaptive_avg_pool1d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        module_op!(
            inputs[(x, float), (grad, float)],
            outputs[out],
            {
                (B::adaptive_avg_pool1d_backward(x, grad),)
            }
        )
        .0
    }

    fn max_pool1d(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        module_op!(inputs[(x, float)], outputs[out], {
            (B::max_pool1d(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ),)
        })
        .0
    }

    fn max_pool1d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> MaxPool1dWithIndices<Self> {
        let (out, indices) = module_op!(
            inputs[(x, float)],
            outputs[(out, Float), (indices, Int)],
            {
                let res = B::max_pool1d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode);
                (res.output, res.indices)
            }
        );
        MaxPool1dWithIndices::new(out, indices)
    }

    fn max_pool1d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool1dBackward<Self> {
        let x_grad = module_op!(
            inputs[(x, float), (output_grad, float), (indices, int)],
            outputs[out],
            {
                let res = B::max_pool1d_with_indices_backward(x, kernel_size, stride, padding, dilation, ceil_mode, output_grad, indices);
                (res.x_grad,)
            }
        )
        .0;
        MaxPool1dBackward::new(x_grad)
    }

    fn attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: Option<burn_backend::tensor::BoolTensor<Self>>,
        attn_bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::AttentionOptions,
    ) -> FloatTensor<Self> {
        // module_op!(
        //     inputs[(queryt, float), (key, float), (value, float)],
        //     opt_inputs[bias], // TODO: bool opt
        //     outputs[out],
        //     {
        //         (B::conv2d(x, weight, bias, options),)
        //     }
        // )
        // .0
        todo!()
    }
}
