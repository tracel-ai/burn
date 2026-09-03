use burn_backend::{
    IntDType,
    ops::{
        DeformConv2dBackward, MaxPool1dBackward, MaxPool1dWithIndices, MaxPool2dBackward,
        MaxPool2dWithIndices, ModuleOps,
    },
    tensor::{FloatTensor, IntTensor},
};
use burn_backend_extension::backend_dispatch;

use crate::Dispatch;

#[backend_dispatch]
impl ModuleOps<Self> for Dispatch {
    fn batch_norm(
        x: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        mean: FloatTensor<Self>,
        variance: FloatTensor<Self>,
        epsilon: f64,
    ) -> FloatTensor<Self> {
        B::batch_norm(x, gamma, beta, mean, variance, epsilon)
    }

    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvOptions<2>,
    ) -> FloatTensor<Self> {
        B::conv2d(x, weight, bias, options)
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        B::deform_conv2d(x, offset, weight, mask, bias, options)
    }

    #[backend_dispatch(skip)]
    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let (x_grad, offset_grad, weight_grad, mask_grad, bias_grad) =
            Self::deform_conv2d_backward_dispatch(
                x,
                offset,
                weight,
                mask,
                bias,
                output_grad,
                options,
            );
        DeformConv2dBackward::new(x_grad, offset_grad, weight_grad, mask_grad, bias_grad)
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvOptions<3>,
    ) -> FloatTensor<Self> {
        B::conv3d(x, weight, bias, options)
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        B::conv_transpose2d(x, weight, bias, options)
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        B::conv_transpose3d(x, weight, bias, options)
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        B::avg_pool2d(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
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
        B::avg_pool2d_backward(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        B::adaptive_avg_pool2d(x, output_size)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        B::adaptive_avg_pool2d_backward(x, grad)
    }

    fn adaptive_avg_pool3d(x: FloatTensor<Self>, output_size: [usize; 3]) -> FloatTensor<Self> {
        B::adaptive_avg_pool3d(x, output_size)
    }

    fn adaptive_avg_pool3d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        B::adaptive_avg_pool3d_backward(x, grad)
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        B::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }

    #[backend_dispatch(skip)]
    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        indices_dtype: IntDType,
    ) -> MaxPool2dWithIndices<Self> {
        let (output, indices) = Self::max_pool2d_with_indices_dispatch(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices_dtype,
        );
        MaxPool2dWithIndices::new(output, indices)
    }

    #[backend_dispatch(skip)]
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
        let x_grad = Self::max_pool2d_with_indices_backward_dispatch(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad,
            indices,
        );
        MaxPool2dBackward::new(x_grad)
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: burn_backend::ops::InterpolateOptions,
    ) -> FloatTensor<Self> {
        B::interpolate(x, output_size, options)
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: burn_backend::ops::InterpolateOptions,
    ) -> FloatTensor<Self> {
        B::interpolate_backward(x, grad, output_size, options)
    }

    fn embedding(weights: FloatTensor<Self>, indices: IntTensor<Self>) -> FloatTensor<Self> {
        B::embedding(weights, indices)
    }

    fn embedding_backward(
        weights: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        B::embedding_backward(weights, output_grad, indices)
    }

    fn conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvOptions<1>,
    ) -> FloatTensor<Self> {
        B::conv1d(x, weight, bias, options)
    }

    fn conv1d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<1>,
    ) -> FloatTensor<Self> {
        B::conv1d_x_backward(x, weight, output_grad, options)
    }

    fn conv1d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<1>,
    ) -> FloatTensor<Self> {
        B::conv1d_weight_backward(x, weight, output_grad, options)
    }

    fn conv1d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        B::conv1d_bias_backward(x, bias, output_grad)
    }

    fn conv2d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<2>,
    ) -> FloatTensor<Self> {
        B::conv2d_x_backward(x, weight, output_grad, options)
    }

    fn conv2d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<2>,
    ) -> FloatTensor<Self> {
        B::conv2d_weight_backward(x, weight, output_grad, options)
    }

    fn conv2d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        B::conv2d_bias_backward(x, bias, output_grad)
    }

    fn conv3d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<3>,
    ) -> FloatTensor<Self> {
        B::conv3d_x_backward(x, weight, output_grad, options)
    }

    fn conv3d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvOptions<3>,
    ) -> FloatTensor<Self> {
        B::conv3d_weight_backward(x, weight, output_grad, options)
    }

    fn conv3d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        B::conv3d_bias_backward(x, bias, output_grad)
    }

    fn conv_transpose1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::ConvTransposeOptions<1>,
    ) -> FloatTensor<Self> {
        B::conv_transpose1d(x, weight, bias, options)
    }

    fn conv_transpose1d_x_backward(
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<1>,
    ) -> FloatTensor<Self> {
        B::conv_transpose1d_x_backward(weight, output_grad, options)
    }

    fn conv_transpose1d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<1>,
    ) -> FloatTensor<Self> {
        B::conv_transpose1d_weight_backward(x, weight, output_grad, options)
    }

    fn conv_transpose1d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        B::conv_transpose1d_bias_backward(x, bias, output_grad)
    }

    fn conv_transpose2d_x_backward(
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        B::conv_transpose2d_x_backward(weight, output_grad, options)
    }

    fn conv_transpose2d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        B::conv_transpose2d_weight_backward(x, weight, output_grad, options)
    }

    fn conv_transpose2d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        B::conv_transpose2d_bias_backward(x, bias, output_grad)
    }

    fn conv_transpose3d_x_backward(
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        B::conv_transpose3d_x_backward(weight, output_grad, options)
    }

    fn conv_transpose3d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        B::conv_transpose3d_weight_backward(x, weight, output_grad, options)
    }

    fn conv_transpose3d_bias_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        B::conv_transpose3d_bias_backward(x, bias, output_grad)
    }

    fn unfold4d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        options: burn_backend::ops::UnfoldOptions,
    ) -> FloatTensor<Self> {
        B::unfold4d(x, kernel_size, options)
    }

    fn avg_pool1d(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        B::avg_pool1d(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
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
        B::avg_pool1d_backward(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn adaptive_avg_pool1d(x: FloatTensor<Self>, output_size: usize) -> FloatTensor<Self> {
        B::adaptive_avg_pool1d(x, output_size)
    }

    fn adaptive_avg_pool1d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        B::adaptive_avg_pool1d_backward(x, grad)
    }

    fn max_pool1d(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        B::max_pool1d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }

    #[backend_dispatch(skip)]
    fn max_pool1d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
        indices_dtype: IntDType,
    ) -> MaxPool1dWithIndices<Self> {
        let (output, indices) = Self::max_pool1d_with_indices_dispatch(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices_dtype,
        );
        MaxPool1dWithIndices::new(output, indices)
    }

    #[backend_dispatch(skip)]
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
        let x_grad = Self::max_pool1d_with_indices_backward_dispatch(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad,
            indices,
        );
        MaxPool1dBackward::new(x_grad)
    }

    fn attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: Option<burn_backend::tensor::BoolTensor<Self>>,
        attn_bias: Option<FloatTensor<Self>>,
        options: burn_backend::ops::AttentionModuleOptions,
    ) -> FloatTensor<Self> {
        B::attention(query, key, value, mask, attn_bias, options)
    }

    fn layer_norm(
        tensor: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        beta: Option<FloatTensor<Self>>,
        epsilon: f64,
    ) -> FloatTensor<Self> {
        B::layer_norm(tensor, gamma, beta, epsilon)
    }

    fn group_norm(
        tensor: FloatTensor<Self>,
        gamma: Option<FloatTensor<Self>>,
        beta: Option<FloatTensor<Self>>,
        num_groups: usize,
        epsilon: f64,
    ) -> FloatTensor<Self> {
        multi_op!(
            inputs[(tensor, float)],
            opt_inputs[(gamma, float), (beta, float)],
            => Float,
            B::group_norm(tensor, gamma, beta, num_groups, epsilon)
        )
    }

    fn rfft(
        signal: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        B::rfft(signal, dim, n)
    }

    fn irfft(
        spectrum_re: FloatTensor<Self>,
        spectrum_im: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Self> {
        B::irfft(spectrum_re, spectrum_im, dim, n)
    }

    #[backend_dispatch(skip)]
    fn has_ctc_loss_backward() -> bool {
        // Dispatch routes per-tensor at runtime, but autodiff queries this flag
        // statically. Returning `false` makes autodiff differentiate through
        // the default decomposed forward, which is safe for every inner
        // backend regardless of whether it has its own ctc_loss_backward.
        false
    }

    fn ctc_loss(
        log_probs: FloatTensor<Self>,
        targets: IntTensor<Self>,
        input_lengths: IntTensor<Self>,
        target_lengths: IntTensor<Self>,
        blank: usize,
    ) -> FloatTensor<Self> {
        B::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank)
    }

    fn ctc_loss_backward(
        log_probs: FloatTensor<Self>,
        targets: IntTensor<Self>,
        input_lengths: IntTensor<Self>,
        target_lengths: IntTensor<Self>,
        grad_loss: FloatTensor<Self>,
        blank: usize,
    ) -> FloatTensor<Self> {
        B::ctc_loss_backward(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            grad_loss,
            blank,
        )
    }

    // TODO: linear ops
    // fn linear(
    //         x: FloatTensor<Self>,
    //         weight: FloatTensor<Self>,
    //         bias: Option<FloatTensor<Self>>,
    //     ) -> FloatTensor<Self> {

    // }
}

#[backend_dispatch]
impl Dispatch {
    fn deform_conv2d_backward_dispatch(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: burn_backend::ops::DeformConvOptions<2>,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        Option<FloatTensor<Self>>,
        Option<FloatTensor<Self>>,
    ) {
        let output = B::deform_conv2d_backward(x, offset, weight, mask, bias, output_grad, options);
        (
            output.x_grad,
            output.offset_grad,
            output.weight_grad,
            output.mask_grad,
            output.bias_grad,
        )
    }

    fn max_pool2d_with_indices_dispatch(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        indices_dtype: IntDType,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        let output = B::max_pool2d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices_dtype,
        );
        (output.output, output.indices)
    }

    fn max_pool2d_with_indices_backward_dispatch(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        B::max_pool2d_with_indices_backward(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad,
            indices,
        )
        .x_grad
    }

    fn max_pool1d_with_indices_dispatch(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
        indices_dtype: IntDType,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        let output = B::max_pool1d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices_dtype,
        );
        (output.output, output.indices)
    }

    fn max_pool1d_with_indices_backward_dispatch(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        B::max_pool1d_with_indices_backward(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad,
            indices,
        )
        .x_grad
    }
}
