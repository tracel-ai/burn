use burn_backend::ops::ModuleOps;
use burn_dispatch::Dispatch;

use crate::{
    Bool, Int, Tensor, TensorPrimitive, check,
    check::TensorCheck,
    ops::{
        AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConvOptions,
        InterpolateOptions, PadMode, PaddedConvOptions, UnfoldOptions,
    },
};

/// Computes the [CTC loss](burn_backend::ops::ModuleOps::ctc_loss).
///
/// # Arguments
///
/// * `log_probs` - Log-probabilities of shape `[T, N, C]`
/// * `targets` - Target label indices of shape `[N, S]`
/// * `input_lengths` - Actual input sequence lengths per batch element `[N]`
/// * `target_lengths` - Actual target lengths per batch element `[N]`
/// * `blank` - Index of the blank label
///
/// # Returns
///
/// Per-sample loss of shape `[N]`
pub fn ctc_loss(
    log_probs: Tensor<3>,
    targets: Tensor<2, Int>,
    input_lengths: Tensor<1, Int>,
    target_lengths: Tensor<1, Int>,
    blank: usize,
) -> Tensor<1> {
    Tensor::new(TensorPrimitive::Float(Dispatch::ctc_loss(
        log_probs.primitive.tensor(),
        targets.primitive,
        input_lengths.primitive,
        target_lengths.primitive,
        blank,
    )))
}

/// Applies the [embedding module](burn_backend::ops::ModuleOps::embedding).
pub fn embedding(weights: Tensor<2>, indices: Tensor<2, Int>) -> Tensor<3> {
    Tensor::new(TensorPrimitive::Float(Dispatch::embedding(
        weights.primitive.tensor(),
        indices.primitive,
    )))
}

/// Applies a [1D convolution](burn_backend::ops::ModuleOps::conv1d).
///
/// Accepts [`ConvOptions`] for symmetric padding, or [`PaddedConvOptions`] for
/// asymmetric padding. When asymmetric padding is specified, an explicit pad
/// operation is applied before the convolution backend op.
pub fn conv1d(
    x: Tensor<3>,
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    options: impl Into<PaddedConvOptions<1>>,
) -> Tensor<3> {
    let padded_options = options.into();
    check!(TensorCheck::conv(
        "conv1d",
        x.dims(),
        weight.dims(),
        padded_options.options.groups,
    ));

    if let Some(padding_end) = padded_options.padding_end {
        let left = padded_options.options.padding[0];
        let right = padding_end[0];
        // For 1D (NCL format), pad the length dimension
        let padded = x.pad((left, right, 0, 0), PadMode::Constant(0.0));
        let zero_options = ConvOptions::new(
            padded_options.options.stride,
            [0],
            padded_options.options.dilation,
            padded_options.options.groups,
        );
        Tensor::new(TensorPrimitive::Float(Dispatch::conv1d(
            padded.primitive.tensor(),
            weight.primitive.tensor(),
            bias.map(|b| b.primitive.tensor()),
            zero_options,
        )))
    } else {
        Tensor::new(TensorPrimitive::Float(Dispatch::conv1d(
            x.primitive.tensor(),
            weight.primitive.tensor(),
            bias.map(|b| b.primitive.tensor()),
            padded_options.options,
        )))
    }
}

/// Applies a [2D convolution](burn_backend::ops::ModuleOps::conv2d).
///
/// Accepts [`ConvOptions`] for symmetric padding, or [`PaddedConvOptions`] for
/// asymmetric padding. When asymmetric padding is specified, an explicit pad
/// operation is applied before the convolution backend op.
pub fn conv2d(
    x: Tensor<4>,
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    options: impl Into<PaddedConvOptions<2>>,
) -> Tensor<4> {
    let padded_options = options.into();
    check!(TensorCheck::conv(
        "conv2d",
        x.dims(),
        weight.dims(),
        padded_options.options.groups,
    ));

    if let Some(padding_end) = padded_options.padding_end {
        let top = padded_options.options.padding[0];
        let left = padded_options.options.padding[1];
        let bottom = padding_end[0];
        let right = padding_end[1];
        // For 2D (NCHW format), pad height and width
        let padded = x.pad((left, right, top, bottom), PadMode::Constant(0.0));
        let zero_options = ConvOptions::new(
            padded_options.options.stride,
            [0, 0],
            padded_options.options.dilation,
            padded_options.options.groups,
        );
        Tensor::new(TensorPrimitive::Float(Dispatch::conv2d(
            padded.primitive.tensor(),
            weight.primitive.tensor(),
            bias.map(|b| b.primitive.tensor()),
            zero_options,
        )))
    } else {
        Tensor::new(TensorPrimitive::Float(Dispatch::conv2d(
            x.primitive.tensor(),
            weight.primitive.tensor(),
            bias.map(|b| b.primitive.tensor()),
            padded_options.options,
        )))
    }
}

/// Applies a [3D convolution](burn_backend::ops::ModuleOps::conv3d).
///
/// Accepts [`ConvOptions`] for symmetric padding, or [`PaddedConvOptions`] for
/// asymmetric padding. Asymmetric 3D padding is not yet supported.
pub fn conv3d(
    x: Tensor<5>,
    weight: Tensor<5>,
    bias: Option<Tensor<1>>,
    options: impl Into<PaddedConvOptions<3>>,
) -> Tensor<5> {
    let padded_options = options.into();
    check!(TensorCheck::conv(
        "conv3d",
        x.dims(),
        weight.dims(),
        padded_options.options.groups,
    ));

    if padded_options.is_asymmetric() {
        panic!("Asymmetric padding is not yet supported for conv3d");
    }

    Tensor::new(TensorPrimitive::Float(Dispatch::conv3d(
        x.primitive.tensor(),
        weight.primitive.tensor(),
        bias.map(|b| b.primitive.tensor()),
        padded_options.options,
    )))
}

/// Applies a [Deformable 2D convolution](burn_backend::ops::ModuleOps::deform_conv2d).
pub fn deform_conv2d(
    x: Tensor<4>,
    offset: Tensor<4>,
    weight: Tensor<4>,
    mask: Option<Tensor<4>>,
    bias: Option<Tensor<1>>,
    options: DeformConvOptions<2>,
) -> Tensor<4> {
    check!(TensorCheck::conv(
        "deform_conv2d",
        x.dims(),
        weight.dims(),
        options.weight_groups,
    ));
    Tensor::new(TensorPrimitive::Float(Dispatch::deform_conv2d(
        x.primitive.tensor(),
        offset.primitive.tensor(),
        weight.primitive.tensor(),
        mask.map(|m| m.primitive.tensor()),
        bias.map(|b| b.primitive.tensor()),
        options,
    )))
}

/// Applies a [1D transposed convolution](burn_backend::ops::ModuleOps::conv_transpose1d).
pub fn conv_transpose1d(
    x: Tensor<3>,
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    options: ConvTransposeOptions<1>,
) -> Tensor<3> {
    check!(TensorCheck::conv_transpose(
        "conv_transpose1d",
        x.dims(),
        weight.dims(),
    ));
    Tensor::new(TensorPrimitive::Float(Dispatch::conv_transpose1d(
        x.primitive.tensor(),
        weight.primitive.tensor(),
        bias.map(|b| b.primitive.tensor()),
        options,
    )))
}

/// Applies a [2D transposed convolution](burn_backend::ops::ModuleOps::conv_transpose2d).
pub fn conv_transpose2d(
    x: Tensor<4>,
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    options: ConvTransposeOptions<2>,
) -> Tensor<4> {
    check!(TensorCheck::conv_transpose(
        "conv_transpose2d",
        x.dims(),
        weight.dims(),
    ));
    Tensor::new(TensorPrimitive::Float(Dispatch::conv_transpose2d(
        x.primitive.tensor(),
        weight.primitive.tensor(),
        bias.map(|b| b.primitive.tensor()),
        options,
    )))
}

/// Applies a 3D transposed convolution](burn_backend::ops::ModuleOps::conv_transpose3d).
pub fn conv_transpose3d(
    x: Tensor<5>,
    weight: Tensor<5>,
    bias: Option<Tensor<1>>,
    options: ConvTransposeOptions<3>,
) -> Tensor<5> {
    check!(TensorCheck::conv_transpose(
        "conv_transpose3d",
        x.dims(),
        weight.dims(),
    ));
    Tensor::new(TensorPrimitive::Float(Dispatch::conv_transpose3d(
        x.primitive.tensor(),
        weight.primitive.tensor(),
        bias.map(|b| b.primitive.tensor()),
        options,
    )))
}

/// Applies a [4D to 3D unfold](burn_backend::ops::ModuleOps::unfold4d).
pub fn unfold4d(x: Tensor<4>, kernel_size: [usize; 2], options: UnfoldOptions) -> Tensor<3> {
    Tensor::new(TensorPrimitive::Float(Dispatch::unfold4d(
        x.primitive.tensor(),
        kernel_size,
        options,
    )))
}

/// Applies a [1D max pooling](burn_backend::ops::ModuleOps::max_pool1d).
pub fn max_pool1d(
    x: Tensor<3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> Tensor<3> {
    Tensor::new(TensorPrimitive::Float(Dispatch::max_pool1d(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )))
}

/// Applies a [2D max pooling](burn_backend::ops::ModuleOps::max_pool2d).
pub fn max_pool2d(
    x: Tensor<4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> Tensor<4> {
    Tensor::new(TensorPrimitive::Float(Dispatch::max_pool2d(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )))
}

/// Applies a [2D avg pooling](burn_backend::ops::ModuleOps::avg_pool2d).
pub fn avg_pool2d(
    x: Tensor<4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> Tensor<4> {
    Tensor::new(TensorPrimitive::Float(Dispatch::avg_pool2d(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        count_include_pad,
        ceil_mode,
    )))
}

/// Applies a [1D avg pooling](burn_backend::ops::ModuleOps::avg_pool1d).
pub fn avg_pool1d(
    x: Tensor<3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
    ceil_mode: bool,
) -> Tensor<3> {
    Tensor::new(TensorPrimitive::Float(Dispatch::avg_pool1d(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        count_include_pad,
        ceil_mode,
    )))
}

/// Applies a [1D max pooling](burn_backend::ops::ModuleOps::max_pool1d).
pub fn max_pool1d_with_indices(
    x: Tensor<3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> (Tensor<3>, Tensor<3, Int>) {
    let output = Dispatch::max_pool1d_with_indices(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    );

    (
        Tensor::new(TensorPrimitive::Float(output.output)),
        Tensor::new(output.indices),
    )
}

/// Applies a [2D max pooling with indices](burn_backend::ops::ModuleOps::max_pool2d_with_indices).
pub fn max_pool2d_with_indices(
    x: Tensor<4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (Tensor<4>, Tensor<4, Int>) {
    let output = Dispatch::max_pool2d_with_indices(
        x.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    );

    (
        Tensor::new(TensorPrimitive::Float(output.output)),
        Tensor::new(output.indices),
    )
}

/// Applies a [2D adaptive avg pooling](burn_backend::ops::ModuleOps::adaptive_avg_pool2d).
pub fn adaptive_avg_pool2d(x: Tensor<4>, output_size: [usize; 2]) -> Tensor<4> {
    Tensor::new(TensorPrimitive::Float(Dispatch::adaptive_avg_pool2d(
        x.primitive.tensor(),
        output_size,
    )))
}

/// Applies a [1D adaptive avg pooling](burn_backend::ops::ModuleOps::adaptive_avg_pool1d).
pub fn adaptive_avg_pool1d(x: Tensor<3>, output_size: usize) -> Tensor<3> {
    Tensor::new(TensorPrimitive::Float(Dispatch::adaptive_avg_pool1d(
        x.primitive.tensor(),
        output_size,
    )))
}

/// Applies a [2D interpolation](burn_backend::ops::ModuleOps::interpolate).
pub fn interpolate(
    x: Tensor<4>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> Tensor<4> {
    Tensor::new(TensorPrimitive::Float(Dispatch::interpolate(
        x.primitive.tensor(),
        output_size,
        options,
    )))
}

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
/// - `bias` is the bias tensor (optional), ``[d_output]``.
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
pub fn linear<const D: usize>(
    input: Tensor<D>,
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
) -> Tensor<D> {
    if D == 1 {
        // Insert and remove an extra batch dimension for the batch matmul to work.
        let input = input.unsqueeze::<2>();
        let output = linear(input, weight, bias);
        return output.squeeze_dim(0);
    }

    Tensor::new(TensorPrimitive::Float(Dispatch::linear(
        input.primitive.tensor(),
        weight.primitive.tensor(),
        bias.map(|b| b.primitive.tensor()),
    )))
}

/// Computes scaled dot-product attention: softmax(QKᵗ * scale) · V,
/// where scale defaults to 1/sqrt(head_dim) (configurable via `options.scale`).
/// Optionally applies masking, additive bias, causal masking, and softcap.
///
/// # Arguments
/// - `query`: Query tensor of shape `[batch_size, num_heads, seq_len_q, head_dim]`
/// - `key`: Key tensor of shape `[batch_size, num_heads, seq_len_k, head_dim]`
/// - `value`: Value tensor of shape `[batch_size, num_heads, seq_len_k, val_dim]`
/// - `mask`: Optional boolean mask of shape `[batch_size, num_heads, seq_len_q, seq_len_k]`,
///   where `true` indicates positions to mask (i.e. set to -inf before softmax).
/// - `attn_bias`: Optional float tensor of shape `[batch_size, num_heads, seq_len_q, seq_len_k]`
///   added to the attention scores before softmax (e.g. ALiBi, relative position biases).
/// - `options`: Additional attention options (custom scale, softcap, causal masking).
///
/// # Returns
/// A tensor of shape `[batch_size, num_heads, seq_len_q, val_dim]`
/// representing the attended context per head.
///
/// # Note
/// This implementation does not support dropout and is intended for inference or
/// use cases where dropout is not needed.
pub fn attention(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    mask: Option<Tensor<4, Bool>>,
    attn_bias: Option<Tensor<4>>,
    options: AttentionModuleOptions,
) -> Tensor<4> {
    Tensor::new(TensorPrimitive::Float(Dispatch::attention(
        query.primitive.tensor(),
        key.primitive.tensor(),
        value.primitive.tensor(),
        mask.map(|mask| mask.primitive),
        attn_bias.map(|bias| bias.primitive.tensor()),
        options,
    )))
}

/// Exports attention fallback to test backend's attention against.
pub fn attention_fallback(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    mask: Option<Tensor<4, Bool>>,
    attn_bias: Option<Tensor<4>>,
    options: AttentionModuleOptions,
) -> Tensor<4> {
    Tensor::new(TensorPrimitive::Float(
        burn_backend::ops::attention::attention_fallback::<Dispatch>(
            query.primitive.tensor(),
            key.primitive.tensor(),
            value.primitive.tensor(),
            mask.map(|mask| mask.primitive),
            attn_bias.map(|bias| bias.primitive.tensor()),
            options,
        ),
    ))
}

/// Calculate the [2D convolution](burn_backend::ops::ModuleOps::conv2d) backward pass, returning the gradient for `weight`.
pub fn conv2d_weight_backward(
    x: Tensor<4>,
    weight: Tensor<4>,
    output_grad: Tensor<4>,
    options: ConvOptions<2>,
) -> Tensor<4> {
    Tensor::new(TensorPrimitive::Float(Dispatch::conv2d_weight_backward(
        x.primitive.tensor(),
        weight.primitive.tensor(),
        output_grad.primitive.tensor(),
        options,
    )))
}

/// Backward pass for the [avg pooling 2d](ModuleOps::avg_pool2d) operation.
pub fn avg_pool2d_backward(
    x: Tensor<4>,
    grad: Tensor<4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> Tensor<4> {
    Tensor::new(TensorPrimitive::Float(Dispatch::avg_pool2d_backward(
        x.primitive.tensor(),
        grad.primitive.tensor(),
        kernel_size,
        stride,
        padding,
        count_include_pad,
        ceil_mode,
    )))
}

/// Backward pass for the [max pooling 2d](ModuleOps::max_pool2d_with_indices) operation.
#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_with_indices_backward(
    x: Tensor<4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    output_grad: Tensor<4>,
    indices: Tensor<4, Int>,
) -> Tensor<4> {
    Tensor::new(TensorPrimitive::Float(
        Dispatch::max_pool2d_with_indices_backward(
            x.primitive.tensor(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad.primitive.tensor(),
            indices.primitive,
        )
        .x_grad,
    ))
}

/// Applies Layer Normalization over the last dimension of the input tensor.
///
/// Computes `(x - mean) / sqrt(var + epsilon) * gamma + beta`, where `mean` and
/// (biased) `var` are reduced over the last axis.
///
/// # Shapes
///
/// - input: `[..., any, d_model]`
/// - output: `[..., any, d_model]`
pub fn layer_norm<const D: usize>(
    input: Tensor<D>,
    gamma: Tensor<1>,
    beta: Option<Tensor<1>>,
    epsilon: f64,
) -> Tensor<D> {
    Tensor::from_primitive(TensorPrimitive::Float(Dispatch::layer_norm(
        input.primitive.tensor(),
        gamma.primitive.tensor(),
        beta.map(|b| b.primitive.tensor()),
        epsilon,
    )))
}
