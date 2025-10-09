use super::{conv, pool};
use crate::ops::unfold::unfold4d_using_conv2d;
use crate::{
    Shape, TensorMetadata,
    backend::Backend,
    ops::{FloatTensor, IntTensor},
};
use alloc::vec;
use core::num::NonZeroUsize;

/// Gradient computed during the backward pass for each tensor used by [conv2d](ModuleOps::conv2d).
#[derive(new)]
pub struct Conv2dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,

    /// Weights gradient.
    pub weights_grad: FloatTensor<B>,

    /// Bias gradient.
    pub bias_grad: Option<FloatTensor<B>>,
}

/// Gradient computed during the backward pass for each tensor used by [deform_conv2d](ModuleOps::deform_conv2d).
#[derive(new)]
pub struct DeformConv2dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,

    /// Offset gradient.
    pub offset_grad: FloatTensor<B>,

    /// Weights gradient.
    pub weight_grad: FloatTensor<B>,

    /// Mask gradient.
    pub mask_grad: Option<FloatTensor<B>>,

    /// Bias gradient.
    pub bias_grad: Option<FloatTensor<B>>,
}

/// Gradient computed during the backward pass for each tensor used by [conv3d](ModuleOps::conv3d).
#[derive(new)]
pub struct Conv3dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,

    /// Weights gradient.
    pub weights_grad: FloatTensor<B>,

    /// Bias gradient.
    pub bias_grad: Option<FloatTensor<B>>,
}

/// Gradient computed during the backward pass for each tensor used by [max_pool1d](ModuleOps::max_pool1d).
#[derive(new)]
pub struct MaxPool1dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,
}

/// Results from [max_pool1d](ModuleOps::max_pool1d_with_indices).
#[derive(new)]
pub struct MaxPool1dWithIndices<B: Backend> {
    /// The output tensor.
    pub output: FloatTensor<B>,

    /// The indices tensor.
    pub indices: IntTensor<B>,
}

/// Gradient computed during the backward pass for each tensor used by [max_pool2d](ModuleOps::max_pool2d).
#[derive(new)]
pub struct MaxPool2dBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,
}

/// Results from [max_pool2d](ModuleOps::max_pool2d_with_indices).
#[derive(new)]
pub struct MaxPool2dWithIndices<B: Backend> {
    /// The output tensor.
    pub output: FloatTensor<B>,

    /// The indices tensor.
    pub indices: IntTensor<B>,
}

/// Check that the parameter value is non-zero.
// NOTE: for now we keep usize but we could refactor the parameters to hold `NonZeroUsize`.
pub(crate) fn check_nonzero(value: usize, msg: &str) -> usize {
    NonZeroUsize::new(value).expect(msg);
    value
}

/// Convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConvOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Groups (non-zero).
    pub groups: usize,
}

impl<const N: usize> ConvOptions<N> {
    /// Constructs a new `ConvOptions`.
    pub fn new(
        stride: [usize; N],
        padding: [usize; N],
        dilation: [usize; N],
        groups: usize,
    ) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
            groups: check_nonzero(groups, "groups must be non-zero"),
        }
    }
}

/// Convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DeformConvOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Weight Groups (non-zero).
    pub weight_groups: usize,

    /// Offset Groups (non-zero).
    pub offset_groups: usize,
}

impl<const N: usize> DeformConvOptions<N> {
    /// Constructs a new `DeformConvOptions`.
    pub fn new(
        stride: [usize; N],
        padding: [usize; N],
        dilation: [usize; N],
        weight_groups: usize,
        offset_groups: usize,
    ) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
            weight_groups: check_nonzero(weight_groups, "weight groups must be non-zero"),
            offset_groups: check_nonzero(offset_groups, "offset groups must be non-zero"),
        }
    }
}

/// Transposed convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConvTransposeOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Padding out.
    pub padding_out: [usize; N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Groups (non-zero).
    pub groups: usize,
}

impl<const N: usize> ConvTransposeOptions<N> {
    /// Constructs a new `ConvTransposeOptions`.
    pub fn new(
        stride: [usize; N],
        padding: [usize; N],
        padding_out: [usize; N],
        dilation: [usize; N],
        groups: usize,
    ) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            padding_out,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
            groups: check_nonzero(groups, "groups must be non-zero"),
        }
    }
}

/// Unfold operation options.
#[derive(Debug, Clone)]
pub struct UnfoldOptions {
    /// The number of positions to slide over the input tensor in each dimension.
    /// A stride of `[1, 1]` will slide the kernel one pixel at a time.
    pub stride: [usize; 2],

    /// The number of zero-padding pixels added to each side of the input tensor in each dimension.
    pub padding: [usize; 2],

    /// The spacing between the blocks (patches) in the original input tensor.
    pub dilation: [usize; 2],
}

impl UnfoldOptions {
    /// Constructs a new `UnfoldOptions`.
    pub fn new(stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2]) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
        }
    }
}

/// Algorithm used for upsampling.
#[derive(new, Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum InterpolateMode {
    /// Nearest-neighbor interpolation.
    /// <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>
    Nearest,

    /// Bilinear interpolation.
    /// <https://en.wikipedia.org/wiki/Bilinear_interpolation>
    Bilinear,

    /// Bicubic interpolation.
    /// <https://en.wikipedia.org/wiki/Bicubic_interpolation>
    Bicubic,
}

/// Interpolation options.
#[derive(new, Debug, Clone)]
pub struct InterpolateOptions {
    /// Algorithm used for upsampling.
    pub mode: InterpolateMode,
}

/// Gradient computed during the backward pass for each tensor used by [interpolate](ModuleOps::interpolate).
#[derive(new)]
pub struct InterpolateBackward<B: Backend> {
    /// Gradient.
    pub x_grad: FloatTensor<B>,
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
    fn embedding(weights: FloatTensor<B>, indices: IntTensor<B>) -> FloatTensor<B> {
        let [batch_size, seq_length] = indices.shape().dims();
        let [_, d_model] = weights.shape().dims();

        let indices = B::int_reshape(indices, Shape::new([batch_size * seq_length]));
        let output = B::float_select(weights, 0, indices);

        B::float_reshape(output, Shape::new([batch_size, seq_length, d_model]))
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
        weights: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        indices: IntTensor<B>,
    ) -> FloatTensor<B> {
        let [batch_size, seq_length] = indices.shape().dims();
        let [n_embeddings, d_model] = weights.shape().dims();
        let device = B::float_device(&weights);
        let dtype = output_grad.dtype();

        let indices = B::int_reshape(indices, Shape::new([batch_size * seq_length]));
        let output_grad =
            B::float_reshape(output_grad, Shape::new([batch_size * seq_length, d_model]));
        let grad = B::float_zeros(Shape::new([n_embeddings, d_model]), &device, dtype.into());

        B::float_select_assign(grad, 0, indices, output_grad)
    }
    /// One dimensional convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, length]`,
    /// weight: `[channels_out, channels_in, kernel_size]`,
    /// bias:   `[channels_out]`,
    fn conv1d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv1d_from_conv2d::<B>(x, weight, bias, options)
    }
    /// Backward pass for the [conv1d](ModuleOps::conv1d) operation, returning the gradient for `x`.
    fn conv1d_x_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv1d_x_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv1d](ModuleOps::conv1d) operation, returning the gradient for `weight`.
    fn conv1d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv1d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv1d](ModuleOps::conv1d) operation, returning the gradient for `bias`.
    fn conv1d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv1d_bias_backward::<B>(x, bias, output_grad)
    }
    /// Two dimensional convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, height, width]`,
    /// weight: `[channels_out, channels_in, kernel_size_1, kernel_size_2]`,
    /// bias:   `[channels_out]`,
    fn conv2d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<B>;
    /// Backward pass for the [conv2d](ModuleOps::conv2d) operation, returning the gradient for `x`.
    fn conv2d_x_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<2>,
    ) -> FloatTensor<B> {
        conv::conv2d_x_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv2d](ModuleOps::conv2d) operation, returning the gradient for `weight`.
    fn conv2d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<2>,
    ) -> FloatTensor<B> {
        conv::conv2d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv2d](ModuleOps::conv2d) operation, returning the gradient for `bias`.
    fn conv2d_bias_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv2d_bias_backward::<B>(x, weight, bias, output_grad)
    }

    /// Two dimensional deformable convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, height, width]`,
    /// weight: `[channels_out, channels_in, kernel_size_1, kernel_size_2]`,
    /// bias:   `[channels_out]`,
    fn deform_conv2d(
        x: FloatTensor<B>,
        offset: FloatTensor<B>,
        weight: FloatTensor<B>,
        mask: Option<FloatTensor<B>>,
        bias: Option<FloatTensor<B>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<B>;
    /// Backward pass for the [deform_conv2d](ModuleOps::deform_conv2d) operation.
    fn deform_conv2d_backward(
        x: FloatTensor<B>,
        offset: FloatTensor<B>,
        weight: FloatTensor<B>,
        mask: Option<FloatTensor<B>>,
        bias: Option<FloatTensor<B>>,
        output_grad: FloatTensor<B>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<B>;

    /// Three dimensional convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, depth, height, width]`,
    /// weight: `[channels_out, channels_in, kernel_size_1, kernel_size_2, kernel_size_3]`,
    /// bias:   `[channels_out]`,
    fn conv3d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<B>;
    /// Backward pass for the [conv3d](ModuleOps::conv3d) operation, returning the gradient for `x`.
    fn conv3d_x_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<3>,
    ) -> FloatTensor<B> {
        conv::conv3d_x_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv3d](ModuleOps::conv3d) operation, returning the gradient for `weight`.
    fn conv3d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<3>,
    ) -> FloatTensor<B> {
        conv::conv3d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv3d](ModuleOps::conv3d) operation, returning the gradient for `bias`.
    fn conv3d_bias_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv3d_bias_backward::<B>(x, weight, bias, output_grad)
    }
    /// One dimensional transposed convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, length]`,
    /// weight: `[channels_in, channels_out, length]`,
    /// bias:   `[channels_out]`,
    fn conv_transpose1d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv_transpose1d_from_conv_transpose2d::<B>(x, weight, bias, options)
    }
    /// Backward pass for the [conv transpose 1d](ModuleOps::conv_transpose1d) operation, returning the gradient for `x`.
    fn conv_transpose1d_x_backward(
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv_transpose1d_x_backward::<B>(weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 1d](ModuleOps::conv_transpose1d) operation, returning the gradient for `weight`.
    fn conv_transpose1d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<B> {
        conv::conv_transpose1d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 1d](ModuleOps::conv_transpose1d) operation, returning the gradient for `bias`.
    fn conv_transpose1d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv_transpose1d_bias_backward::<B>(x, bias, output_grad)
    }

    /// Two dimensional transposed convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, height, width]`,
    /// weight: `[channels_in, channels_out, kernel_size_1, kernel_size_2]`,
    /// bias:   `[channels_out]`,
    fn conv_transpose2d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<B>;
    /// Backward pass for the [conv transpose 2d](ModuleOps::conv_transpose2d) operation, returning the gradient for `x`.
    fn conv_transpose2d_x_backward(
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<B> {
        conv::conv_transpose2d_x_backward::<B>(weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 2d](ModuleOps::conv_transpose2d) operation, returning the gradient for `weight`.
    fn conv_transpose2d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<B> {
        conv::conv_transpose2d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 2d](ModuleOps::conv_transpose2d) operation, returning the gradient for `bias`.
    fn conv_transpose2d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv_transpose2d_bias_backward::<B>(x, bias, output_grad)
    }

    /// Three dimensional transposed convolution.
    ///
    /// # Shapes
    ///
    /// x:      `[batch_size, channels_in, height, width]`,
    /// weight: `[channels_in, channels_out, kernel_size_1, kernel_size_2, kernel_size_3]`,
    /// bias:   `[channels_out]`,
    fn conv_transpose3d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<B>;
    /// Backward pass for the [conv transpose 3d](ModuleOps::conv_transpose3d) operation, returning the gradient for `x`.
    fn conv_transpose3d_x_backward(
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<B> {
        conv::conv_transpose3d_x_backward::<B>(weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 3d](ModuleOps::conv_transpose3d) operation, returning the gradient for `weight`.
    fn conv_transpose3d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<B> {
        conv::conv_transpose3d_weight_backward::<B>(x, weight, output_grad, options)
    }
    /// Backward pass for the [conv transpose 3d](ModuleOps::conv_transpose3d) operation, returning the gradient for `bias`.
    fn conv_transpose3d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        conv::conv_transpose3d_bias_backward::<B>(x, bias, output_grad)
    }

    /// Four-dimensional unfolding.
    ///
    /// # Shapes
    ///
    /// * x:      ``[batch_size, channels_in, height, width]``,
    /// * returns: ``[batch_size, channels_in * kernel_size_1 * kernel_size_2, number of blocks]``,
    fn unfold4d(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        options: UnfoldOptions,
    ) -> FloatTensor<B> {
        if options.padding == [0, 0] && options.dilation == [1, 1] {
            let blocks = B::float_unfold(x, 2, kernel_size[0], options.stride[0]);
            let blocks = B::float_unfold(blocks, 3, kernel_size[1], options.stride[1]);

            // batch, channels, h_blocks, w_blocks, h_kern, w_kern

            let blocks = B::float_permute(blocks, &[0, 1, 4, 5, 2, 3]);
            let shape = &blocks.shape().dims;

            // batch, channels, h_kern, w_kern, h_blocks, w_blocks

            B::float_reshape(
                blocks,
                [
                    shape[0],
                    shape[1] * shape[2] * shape[3],
                    shape[4] * shape[5],
                ]
                .into(),
            )
        } else {
            unfold4d_using_conv2d::<B>(x, kernel_size, options)
        }
    }

    /// One dimensional avg pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, length],
    fn avg_pool1d(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<B> {
        pool::avg_pool1d_from_2d::<B>(x, kernel_size, stride, padding, count_include_pad)
    }
    /// Backward pass for the [avg pooling 1d](ModuleOps::avg_pool1d) operation.
    fn avg_pool1d_backward(
        x: FloatTensor<B>,
        grad: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<B> {
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
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<B>;
    /// Backward pass for the [avg pooling 2d](ModuleOps::avg_pool2d) operation.
    fn avg_pool2d_backward(
        x: FloatTensor<B>,
        grad: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<B>;
    /// Two dimensional adaptive avg pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn adaptive_avg_pool2d(x: FloatTensor<B>, output_size: [usize; 2]) -> FloatTensor<B>;
    /// Backward pass for the [adaptive avg pooling 2d](ModuleOps::adaptive_avg_pool2d) operation.
    fn adaptive_avg_pool2d_backward(x: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B>;
    /// One dimensional adaptive avg pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, length],
    fn adaptive_avg_pool1d(x: FloatTensor<B>, output_size: usize) -> FloatTensor<B> {
        pool::adaptive_avg_pool1d_from_2d::<B>(x, output_size)
    }
    /// Backward pass for the [adaptive avg pooling 1d](ModuleOps::adaptive_avg_pool1d) operation.
    fn adaptive_avg_pool1d_backward(x: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B> {
        pool::adaptive_avg_pool1d_backward_from_2d::<B>(x, grad)
    }
    /// One dimensional max pooling.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, length],
    fn max_pool1d(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> FloatTensor<B> {
        pool::max_pool1d_from_2d::<B>(x, kernel_size, stride, padding, dilation)
    }

    /// One dimensional max pooling with indices.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn max_pool1d_with_indices(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<B> {
        pool::max_pool1d_with_indices_from_2d::<B>(x, kernel_size, stride, padding, dilation)
    }
    /// Backward pass for the [max pooling 1d](ModuleOps::max_pool1d_with_indices) operation.
    fn max_pool1d_with_indices_backward(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output_grad: FloatTensor<B>,
        indices: IntTensor<B>,
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
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<B>;

    /// Two dimensional max pooling with indices.
    ///
    /// # Shapes
    ///
    /// x: [batch_size, channels, height, width],
    fn max_pool2d_with_indices(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<B>;
    /// Backward pass for the [max pooling 2d](ModuleOps::max_pool2d_with_indices) operation.
    fn max_pool2d_with_indices_backward(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: FloatTensor<B>,
        indices: IntTensor<B>,
    ) -> MaxPool2dBackward<B>;

    /// Down/up samples the input.
    ///
    /// # Shapes
    ///
    /// x: `[batch_size, channels, height, width]`,
    fn interpolate(
        x: FloatTensor<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<B>;

    /// Backward pass for the [interpolate](ModuleOps::interpolate) operation.
    fn interpolate_backward(
        x: FloatTensor<B>,
        grad: FloatTensor<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<B>;

    /// Applies a linear transformation to the input tensor using the given weight and bias.
    ///
    /// ```math
    /// y = x @ weight + [bias]
    /// ```
    ///
    /// # Arguments:
    ///
    /// - `input` is the input tensor, ``[..., d_input]``.
    /// - `weight` is the weight tensor, ``[d_input, d_output]``.
    /// - `b` is the bias tensor (optional), ``[d_output]``.
    ///
    /// # Returns:
    ///
    /// The transformed tensor, ``[..., d_output]``.
    ///
    /// # Compatibility
    ///
    /// This function differs from PyTorch's ``torch.nn.functional.linear`` in that it does not
    /// transpose the weight matrix. In PyTorch, the weight matrix is transposed before
    /// multiplication:
    ///
    /// ```math
    /// y = x @ weight^T + [bias]
    /// ```
    fn linear(
        input: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
    ) -> FloatTensor<B> {
        let ndims_in = input.shape().num_dims();
        let [d_input, d_output] = weight.shape().dims();

        if ndims_in == 1 {
            // Insert and remove an extra batch dimension for the batch matmul to work.
            let input = B::float_reshape(input, Shape::from([1, d_input]));
            let output = Self::linear(input, weight, bias);
            return B::float_reshape(output, Shape::from([d_output]));
        }

        // Perform broadcasting
        //
        // Important to be done before doing operations to easily fuse.
        let weight = unsqueeze::<B>(weight, ndims_in);
        let bias = bias.map(|bias| unsqueeze::<B>(bias, ndims_in));

        let output = B::float_matmul(input, weight);
        match bias {
            Some(bias) => B::float_add(output, bias),
            None => output,
        }
    }
}

// Unsqueeze op on primitive.
// TODO: would be nice to have this on primitives too for convenience.
fn unsqueeze<B: Backend>(tensor: FloatTensor<B>, ndims_out: usize) -> FloatTensor<B> {
    let shape = tensor.shape();
    let ndims_in = shape.num_dims();

    let mut dims = vec![1; ndims_out];
    let num_ones = ndims_out - ndims_in;
    dims[num_ones..(ndims_in + num_ones)].copy_from_slice(&shape[..ndims_in]);

    B::float_reshape(tensor, Shape::from(dims))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn conv_options_stride_zero() {
        let _opt = ConvOptions::new([0, 1], [0, 0], [1, 1], 1);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn conv_options_dilation_zero() {
        let _opt = ConvOptions::new([1, 1], [0, 0], [0, 0], 1);
    }

    #[test]
    #[should_panic = "groups must be non-zero"]
    fn conv_options_groups_zero() {
        let _opt = ConvOptions::new([1, 1], [0, 0], [1, 1], 0);
    }

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn conv_transpose_options_stride_zero() {
        let _opt = ConvTransposeOptions::new([0, 1], [0, 0], [0, 0], [1, 1], 1);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn conv_transpose_options_dilation_zero() {
        let _opt = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [0, 0], 1);
    }

    #[test]
    #[should_panic = "groups must be non-zero"]
    fn conv_transpose_options_groups_zero() {
        let _opt = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], 0);
    }

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn deform_conv_options_stride_zero() {
        let _opt = DeformConvOptions::new([0, 1], [0, 0], [1, 1], 1, 1);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn deform_conv_options_dilation_zero() {
        let _opt = DeformConvOptions::new([1, 1], [0, 0], [0, 0], 1, 1);
    }

    #[test]
    #[should_panic = "weight groups must be non-zero"]
    fn deform_conv_options_weights_groups_zero() {
        let _opt = DeformConvOptions::new([1, 1], [0, 0], [1, 1], 0, 1);
    }

    #[test]
    #[should_panic = "offset groups must be non-zero"]
    fn deform_conv_options_offset_groups_zero() {
        let _opt = DeformConvOptions::new([1, 1], [0, 0], [1, 1], 1, 0);
    }

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn unfold_options_stride_zero() {
        let _opt = UnfoldOptions::new([0, 1], [0, 0], [1, 1]);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn unfold_options_dilation_zero() {
        let _opt = UnfoldOptions::new([1, 1], [0, 0], [0, 0]);
    }
}
