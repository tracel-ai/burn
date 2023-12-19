use super::{conv, pool, unfold::unfold4d_using_conv2d};
use crate::{
    backend::Backend,
    ops::{FloatTensor, IntTensor},
    Shape,
};

/// Gradient computed during the backward pass for each tensor used by [conv2d](ModuleOps::conv2d).
#[derive(new)]
pub struct Conv2dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B, 4>,

    /// Weights gradient.
    pub weights_grad: FloatTensor<B, 4>,

    /// Bias gradient.
    pub bias_grad: Option<FloatTensor<B, 1>>,
}

/// Gradient computed during the backward pass for each tensor used by [max_pool1d](ModuleOps::max_pool1d).
#[derive(new)]
pub struct MaxPool1dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B, 3>,
}

/// Results from [max_pool1d](ModuleOps::max_pool1d_with_indices).
#[derive(new)]
pub struct MaxPool1dWithIndices<B: Backend> {
    /// The output tensor.
    pub output: FloatTensor<B, 3>,

    /// The indices tensor.
    pub indices: IntTensor<B, 3>,
}

/// Gradient computed during the backward pass for each tensor used by [max_pool2d](ModuleOps::max_pool2d).
#[derive(new)]
pub struct MaxPool2dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B, 4>,
}

/// Results from [max_pool2d](ModuleOps::max_pool2d_with_indices).
#[derive(new)]
pub struct MaxPool2dWithIndices<B: Backend> {
    /// The output tensor.
    pub output: FloatTensor<B, 4>,

    /// The indices tensor.
    pub indices: IntTensor<B, 4>,
}

/// Gradient computed during the backward pass for each tensor used by [conv1d](ModuleOps::conv1d).
#[derive(new)]
pub struct Conv1dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B, 3>,

    /// Weights gradient.
    pub weights_grad: FloatTensor<B, 3>,

    /// Bias gradient.
    pub bias_grad: Option<FloatTensor<B, 1>>,
}

/// Convolution options.
#[derive(new, Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConvOptions<const N: usize> {
    /// Stride.
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Dilation.
    pub dilation: [usize; N],

    /// Groups.
    pub groups: usize,
}

/// Transposed convolution options.
#[derive(new, Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConvTransposeOptions<const N: usize> {
    /// Stride.
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Padding out.
    pub padding_out: [usize; N],

    /// Dilation.
    pub dilation: [usize; N],

    /// Groups.
    pub groups: usize,
}

/// Unfold operation options.
#[derive(new, Debug, Clone)]
pub struct UnfoldOptions {
    /// The number of positions to slide over the input tensor in each dimension.
    /// A stride of `[1, 1]` will slide the kernel one pixel at a time.
    pub stride: [usize; 2],

    /// The number of zero-padding pixels added to each side of the input tensor in each dimension.
    pub padding: [usize; 2],

    /// The spacing between the blocks (patches) in the original input tensor.
    pub dilation: [usize; 2],
}

/// Module operations trait.
pub trait ModuleOps<B: Backend> {
    /// Embedding operation.
    ///
    /// # Arguments
    ///
    /// * `weights` - The embedding weights.
    /// * `indices` - The indices tensor.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn embedding(weights: FloatTensor<B, 2>, indices: IntTensor<B, 2>) -> FloatTensor<B, 3> {
        let [batch_size, seq_length] = B::int_shape(&indices).dims;
        let [_, d_model] = B::shape(&weights).dims;

        let indices = B::int_reshape(indices, Shape::new([batch_size * seq_length]));
        let output = B::select(weights, 0, indices);

        B::reshape(output, Shape::new([batch_size, seq_length, d_model]))
    }

    /// Embedding backward operation.
    ///
    /// # Arguments
    ///
    /// * `weights` - The embedding weights.
    /// * `output_grad` - The output gradient.
    /// * `indices` - The indices tensor.
    ///
    /// # Returns
    ///
    /// The gradient.
    fn embedding_backward(
        weights: FloatTensor<B, 2>,
        output_grad: FloatTensor<B, 3>,
        indices: IntTensor<B, 2>,
    ) -> FloatTensor<B, 2> {
        let [batch_size, seq_length] = B::int_shape(&indices).dims;
        let [n_embeddings, d_model] = B::shape(&weights).dims;
        let device = B::device(&weights);

        let indices = B::int_reshape(indices, Shape::new([batch_size * seq_length]));
        let output_grad = B::reshape(output_grad, Shape::new([batch_size * seq_length, d_model]));
        let grad = B::zeros(Shape::new([n_embeddings, d_model]), &device);

        B::select_assign(grad, 0, indices, output_grad)
    }
    /// One dimensional convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, length]`,
    /// weight: `[channels_out, channels_in, kernel_size]`,
    /// bias:   `[channels_out]`,
    fn conv1d(
        x: FloatTensor<B, 3>,
        weight: FloatTensor<B, 3>,
        bias: Option<FloatTensor<B, 1>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<B, 3> {
        conv::conv1d_from_conv2d::<B>(x, weight, bias, options)
    }
    /// Backward pass for the [conv1d](ModuleOps::conv1d) operation.
    fn conv1d_backward(
        x: FloatTensor<B, 3>,
        weight: FloatTensor<B, 3>,
        bias: Option<FloatTensor<B, 1>>,
        output_grad: FloatTensor<B, 3>,
        options: ConvOptions<1>,
    ) -> Conv1dBackward<B> {
        conv::conv1d_backward(x, weight, bias, output_grad, options)
    }
    /// Two dimensional convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, height, width]`,
    /// weight: `[channels_out, channels_in, kernel_size_1, kernel_size_2]`,
    /// bias:   `[channels_out]`,
    fn conv2d(
        x: FloatTensor<B, 4>,
        weight: FloatTensor<B, 4>,
        bias: Option<FloatTensor<B, 1>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<B, 4>;
    /// Backward pass for the [conv2d](ModuleOps::conv2d) operation.
    fn conv2d_backward(
        x: FloatTensor<B, 4>,
        weight: FloatTensor<B, 4>,
        bias: Option<FloatTensor<B, 1>>,
        output_grad: FloatTensor<B, 4>,
        options: ConvOptions<2>,
    ) -> Conv2dBackward<B> {
        conv::conv2d_backward(x, weight, bias, output_grad, options)
    }
    /// One dimensional transposed convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, length]`,
    /// weight: `[channels_in, channels_out, length]`,
    /// bias:   `[channels_out]`,
    fn conv_transpose1d(
        x: FloatTensor<B, 3>,
        weight: FloatTensor<B, 3>,
        bias: Option<FloatTensor<B, 1>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<B, 3> {
        conv::conv_transpose1d_from_conv_transpose2d::<B>(x, weight, bias, options)
    }
    /// Backward pass for the [conv transpose 1d](ModuleOps::conv_transpose1d) operation.
    fn conv_transpose1d_backward(
        x: FloatTensor<B, 3>,
        weight: FloatTensor<B, 3>,
        bias: Option<FloatTensor<B, 1>>,
        output_grad: FloatTensor<B, 3>,
        options: ConvTransposeOptions<1>,
    ) -> Conv1dBackward<B> {
        conv::conv_transpose1d_backward(x, weight, bias, output_grad, options)
    }
    /// Two dimensional transposed convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, height, width]`,
    /// weight: `[channels_in, channels_out, kernel_size_1, kernel_size_2]`,
    /// bias:   `[channels_out]`,
    fn conv_transpose2d(
        x: FloatTensor<B, 4>,
        weight: FloatTensor<B, 4>,
        bias: Option<FloatTensor<B, 1>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<B, 4>;

    /// Backward pass for the [conv transpose 2d](ModuleOps::conv_transpose2d) operation.
    fn conv_transpose2d_backward(
        x: FloatTensor<B, 4>,
        weight: FloatTensor<B, 4>,
        bias: Option<FloatTensor<B, 1>>,
        output_grad: FloatTensor<B, 4>,
        options: ConvTransposeOptions<2>,
    ) -> Conv2dBackward<B> {
        conv::conv_transpose2d_backward(x, weight, bias, output_grad, options)
    }

    /// Four-dimensional unfolding.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, height, width]`,
    /// returns: `[batch_size, channels_in * kernel_size_1 * kernel_size_2, number of blocks]`,
    fn unfold4d(
        x: FloatTensor<B, 4>,
        kernel_size: [usize; 2],
        options: UnfoldOptions,
    ) -> FloatTensor<B, 3> {
        unfold4d_using_conv2d::<B>(x, kernel_size, options)
    }

    /// One dimensional avg pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, length],
    fn avg_pool1d(
        x: FloatTensor<B, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<B, 3> {
        pool::avg_pool1d_from_2d::<B>(x, kernel_size, stride, padding, count_include_pad)
    }
    /// Backward pass for the [avg pooling 1d](ModuleOps::avg_pool1d) operation.
    fn avg_pool1d_backward(
        x: FloatTensor<B, 3>,
        grad: FloatTensor<B, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<B, 3> {
        pool::avg_pool1d_backward_from_2d::<B>(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
        )
    }
    /// Two dimensional avg pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn avg_pool2d(
        x: FloatTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<B, 4>;
    /// Backward pass for the [avg pooling 2d](ModuleOps::avg_pool2d) operation.
    fn avg_pool2d_backward(
        x: FloatTensor<B, 4>,
        grad: FloatTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<B, 4>;
    /// Two dimensional adaptive avg pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn adaptive_avg_pool2d(x: FloatTensor<B, 4>, output_size: [usize; 2]) -> FloatTensor<B, 4>;
    /// Backward pass for the [adaptive avg pooling 2d](ModuleOps::adaptive_avg_pool2d) operation.
    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<B, 4>,
        grad: FloatTensor<B, 4>,
    ) -> FloatTensor<B, 4>;
    /// One dimensional adaptive avg pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, length],
    fn adaptive_avg_pool1d(x: FloatTensor<B, 3>, output_size: usize) -> FloatTensor<B, 3> {
        pool::adaptive_avg_pool1d_from_2d::<B>(x, output_size)
    }
    /// Backward pass for the [adaptive avg pooling 1d](ModuleOps::adaptive_avg_pool1d) operation.
    fn adaptive_avg_pool1d_backward(
        x: FloatTensor<B, 3>,
        grad: FloatTensor<B, 3>,
    ) -> FloatTensor<B, 3> {
        pool::adaptive_avg_pool1d_backward_from_2d::<B>(x, grad)
    }
    /// One dimensional max pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, length],
    fn max_pool1d(
        x: FloatTensor<B, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> FloatTensor<B, 3> {
        pool::max_pool1d_from_2d::<B>(x, kernel_size, stride, padding, dilation)
    }

    /// One dimensional max pooling with indices.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn max_pool1d_with_indices(
        x: FloatTensor<B, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<B> {
        pool::max_pool1d_with_indices_from_2d::<B>(x, kernel_size, stride, padding, dilation)
    }
    /// Backward pass for the [max pooling 1d](ModuleOps::max_pool1d_with_indices) operation.
    fn max_pool1d_with_indices_backward(
        x: FloatTensor<B, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output_grad: FloatTensor<B, 3>,
        indices: IntTensor<B, 3>,
    ) -> MaxPool1dBackward<B> {
        pool::max_pool1d_with_indices_backward_from_2d::<B>(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            output_grad,
            indices,
        )
    }

    /// Two dimensional max pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn max_pool2d(
        x: FloatTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<B, 4>;

    /// Two dimensional max pooling with indices.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn max_pool2d_with_indices(
        x: FloatTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<B>;
    /// Backward pass for the [max pooling 2d](ModuleOps::max_pool2d_with_indices) operation.
    fn max_pool2d_with_indices_backward(
        x: FloatTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: FloatTensor<B, 4>,
        indices: IntTensor<B, 4>,
    ) -> MaxPool2dBackward<B>;
}
