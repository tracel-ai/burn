use burn_backend::ops::ModuleOps;
use burn_dispatch::Dispatch;
use burn_std::{MatmulTransformAction, MatmulTransformAnalysis, MatmulTransformPolicy};

use crate::{
    Bool, DType, Int, Tensor, check,
    check::TensorCheck,
    kind::Basic,
    ops::{
        AttentionModuleOptions, AvgPoolOptions, BridgeTensor, ConvOptions, ConvTransposeOptions,
        DeformConvOptions, InterpolateOptions, MaxPoolOptions, PadMode, UnfoldOptions,
    },
};

/// Applies batch normalization using explicitly supplied channel statistics.
///
/// `input` has shape `[batch, channels, ...]`; `gamma`, `beta`, `mean`, and
/// `variance` each have shape `[channels]`.
///
/// This function doesn't calculate or update statistics. Callers may supply
/// running statistics for inference or batch statistics calculated by a
/// training path.
pub fn batch_norm<const D: usize>(
    input: Tensor<D>,
    gamma: Tensor<1>,
    beta: Tensor<1>,
    mean: Tensor<1>,
    variance: Tensor<1>,
    epsilon: f64,
) -> Tensor<D> {
    assert!(D >= 2, "batch norm requires an input rank of at least 2");
    let channels = input.dims()[1];
    assert_eq!(gamma.dims(), [channels], "invalid batch norm gamma shape");
    assert_eq!(beta.dims(), [channels], "invalid batch norm beta shape");
    assert_eq!(mean.dims(), [channels], "invalid batch norm mean shape");
    assert_eq!(
        variance.dims(),
        [channels],
        "invalid batch norm variance shape"
    );
    Tensor::new(BridgeTensor::float(Dispatch::batch_norm(
        input.primitive.into_float(),
        gamma.primitive.into_float(),
        beta.primitive.into_float(),
        mean.primitive.into_float(),
        variance.primitive.into_float(),
        epsilon,
    )))
}

/// Output and batch statistics from [`batch_norm_train`].
pub struct BatchNormTrainOutput<const D: usize> {
    /// The normalized input, with the same shape as the input.
    pub output: Tensor<D>,

    /// The batch mean with shape `[channels]`, detached on autodiff devices.
    pub mean: Tensor<1>,

    /// The biased (population) batch variance with shape `[channels]`, excluding
    /// epsilon and detached on autodiff devices.
    pub variance: Tensor<1>,
}

/// Applies batch normalization using statistics computed from the input batch.
///
/// `input` has shape `[batch, channels, ...]`; `gamma` and `beta` have shape
/// `[channels]`.
///
/// Returns the normalized input with its original shape, along with the batch
/// mean and biased (population) variance, both with shape `[channels]`.
/// Statistics are computed over every dimension except the channel dimension.
/// The returned variance excludes `epsilon`.
///
/// This function does not update running statistics. For normalization using
/// explicitly supplied statistics, use [`batch_norm`].
///
/// # Autodiff
///
/// This function can be used with or without autodiff. It does not enable
/// gradient tracking.
///
/// On an autodiff-enabled device, gradients through the normalized output
/// account for the dependence of the batch statistics on the input. The returned
/// mean and variance remain on the same device but are detached.
pub fn batch_norm_train<const D: usize>(
    input: Tensor<D>,
    gamma: Tensor<1>,
    beta: Tensor<1>,
    epsilon: f64,
) -> BatchNormTrainOutput<D> {
    assert!(D >= 2, "batch norm requires an input rank of at least 2");
    let channels = input.dims()[1];
    assert_eq!(gamma.dims(), [channels], "invalid batch norm gamma shape");
    assert_eq!(beta.dims(), [channels], "invalid batch norm beta shape");
    let result = Dispatch::batch_norm_train(
        input.primitive.into_float(),
        gamma.primitive.into_float(),
        beta.primitive.into_float(),
        epsilon,
    );

    BatchNormTrainOutput {
        output: Tensor::new(BridgeTensor::float(result.output)),
        mean: Tensor::new(BridgeTensor::float(result.mean)),
        variance: Tensor::new(BridgeTensor::float(result.variance)),
    }
}

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
    Tensor::new(BridgeTensor::float(Dispatch::ctc_loss(
        log_probs.primitive.into_float(),
        targets.primitive.into(),
        input_lengths.primitive.into(),
        target_lengths.primitive.into(),
        blank,
    )))
}

/// Applies the [embedding module](burn_backend::ops::ModuleOps::embedding).
pub fn embedding(weights: Tensor<2>, indices: Tensor<2, Int>) -> Tensor<3> {
    Tensor::new(BridgeTensor::float(Dispatch::embedding(
        weights.primitive.into_float(),
        indices.primitive.into(),
    )))
}

/// Applies a [1D convolution](burn_backend::ops::ModuleOps::conv1d).
///
/// Supports symmetric and asymmetric padding through [`ConvOptions`].
/// The deprecated [`PaddedConvOptions`](crate::ops::PaddedConvOptions) is also
/// accepted for compatibility.
pub fn conv1d(
    x: Tensor<3>,
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    options: impl Into<ConvOptions<1>>,
) -> Tensor<3> {
    let options = options.into();
    check!(TensorCheck::conv(
        "conv1d",
        x.dims(),
        weight.dims(),
        options.groups,
    ));

    Tensor::new(BridgeTensor::float(Dispatch::conv1d(
        x.primitive.into_float(),
        weight.primitive.into_float(),
        bias.map(|b| b.primitive.into_float()),
        options,
    )))
}

/// Applies a [2D convolution](burn_backend::ops::ModuleOps::conv2d).
///
/// Supports symmetric and asymmetric padding through [`ConvOptions`].
/// The deprecated [`PaddedConvOptions`](crate::ops::PaddedConvOptions) is also
/// accepted for compatibility.
pub fn conv2d(
    x: Tensor<4>,
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    options: impl Into<ConvOptions<2>>,
) -> Tensor<4> {
    let options = options.into();
    check!(TensorCheck::conv(
        "conv2d",
        x.dims(),
        weight.dims(),
        options.groups,
    ));

    Tensor::new(BridgeTensor::float(Dispatch::conv2d(
        x.primitive.into_float(),
        weight.primitive.into_float(),
        bias.map(|b| b.primitive.into_float()),
        options,
    )))
}

/// Applies a [3D convolution](burn_backend::ops::ModuleOps::conv3d).
///
/// Asymmetric 3D padding is not yet supported.
/// The deprecated [`PaddedConvOptions`](crate::ops::PaddedConvOptions) is also
/// accepted for compatibility.
pub fn conv3d(
    x: Tensor<5>,
    weight: Tensor<5>,
    bias: Option<Tensor<1>>,
    options: impl Into<ConvOptions<3>>,
) -> Tensor<5> {
    let options = options.into();
    check!(TensorCheck::conv(
        "conv3d",
        x.dims(),
        weight.dims(),
        options.groups,
    ));

    if options.is_asymmetric() {
        panic!("Asymmetric padding is not yet supported for conv3d");
    }

    Tensor::new(BridgeTensor::float(Dispatch::conv3d(
        x.primitive.into_float(),
        weight.primitive.into_float(),
        bias.map(|b| b.primitive.into_float()),
        options,
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
    Tensor::new(BridgeTensor::float(Dispatch::deform_conv2d(
        x.primitive.into_float(),
        offset.primitive.into_float(),
        weight.primitive.into_float(),
        mask.map(|m| m.primitive.into_float()),
        bias.map(|b| b.primitive.into_float()),
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
    Tensor::new(BridgeTensor::float(Dispatch::conv_transpose1d(
        x.primitive.into_float(),
        weight.primitive.into_float(),
        bias.map(|b| b.primitive.into_float()),
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
    Tensor::new(BridgeTensor::float(Dispatch::conv_transpose2d(
        x.primitive.into_float(),
        weight.primitive.into_float(),
        bias.map(|b| b.primitive.into_float()),
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
    Tensor::new(BridgeTensor::float(Dispatch::conv_transpose3d(
        x.primitive.into_float(),
        weight.primitive.into_float(),
        bias.map(|b| b.primitive.into_float()),
        options,
    )))
}

/// Applies a [4D to 3D unfold](burn_backend::ops::ModuleOps::unfold4d).
pub fn unfold4d(x: Tensor<4>, kernel_size: [usize; 2], options: UnfoldOptions) -> Tensor<3> {
    Tensor::new(BridgeTensor::float(Dispatch::unfold4d(
        x.primitive.into_float(),
        kernel_size,
        options,
    )))
}

/// Applies a 3D to 4D fold, the inverse of [unfold4d].
///
/// Combines an array of sliding local blocks into a large containing tensor, summing the
/// values of blocks that overlap. This is the operation performed by
/// [`torch.nn.Fold`](https://pytorch.org/docs/stable/generated/torch.nn.Fold.html), and is the
/// adjoint of [unfold4d]: it reuses the same one-hot kernel through a [conv_transpose2d].
///
/// # Arguments
///
/// * `x` - Input columns of shape
///   `[batch_size, channels * kernel_size_0 * kernel_size_1, number_of_blocks]`.
/// * `output_size` - The spatial size `[height, width]` of the folded output tensor.
/// * `kernel_size` - The size of the sliding blocks.
/// * `options` - The stride, padding and dilation of the matching unfold.
///
/// # Returns
///
/// A tensor of shape `[batch_size, channels, output_size_0, output_size_1]`.
pub fn fold4d(
    x: Tensor<3>,
    output_size: [usize; 2],
    kernel_size: [usize; 2],
    options: UnfoldOptions,
) -> Tensor<4> {
    Tensor::new(BridgeTensor::float(Dispatch::fold4d(
        x.primitive.into_float(),
        output_size,
        kernel_size,
        options,
    )))
}

/// Applies a [1D max pooling](burn_backend::ops::ModuleOps::max_pool1d).
///
/// Supports symmetric and asymmetric padding through [`MaxPoolOptions`].
pub fn max_pool1d(x: Tensor<3>, options: MaxPoolOptions<1>) -> Tensor<3> {
    let dims = x.dims();
    let (x, padding) = pad_max_pool_input(x, &options);
    let output = Tensor::new(BridgeTensor::float(Dispatch::max_pool1d(
        x.primitive.into_float(),
        options.kernel_size[0],
        options.stride[0],
        padding[0],
        options.dilation[0],
        options.ceil_mode,
    )));

    drop_end_padding_windows(
        output,
        dims,
        options.stride,
        options.padding,
        options.ceil_mode,
    )
}

/// Applies a [2D max pooling](burn_backend::ops::ModuleOps::max_pool2d).
///
/// Supports symmetric and asymmetric padding through [`MaxPoolOptions`].
pub fn max_pool2d(x: Tensor<4>, options: MaxPoolOptions<2>) -> Tensor<4> {
    let dims = x.dims();
    let (x, padding) = pad_max_pool_input(x, &options);
    let output = Tensor::new(BridgeTensor::float(Dispatch::max_pool2d(
        x.primitive.into_float(),
        options.kernel_size,
        options.stride,
        padding,
        options.dilation,
        options.ceil_mode,
    )));

    drop_end_padding_windows(
        output,
        dims,
        options.stride,
        options.padding,
        options.ceil_mode,
    )
}

/// Applies a [3D max pooling](burn_backend::ops::ModuleOps::max_pool3d).
///
/// Supports symmetric and asymmetric padding through [`MaxPoolOptions`].
///
/// # Panics
///
/// - If any dimension of `kernel_size` is 0.
/// - If any dimension of `stride` is 0.
/// - If any dimension of `dilation` is 0.
/// - If any dimension of `padding` exceeds `kernel_size / 2`.
pub fn max_pool3d(x: Tensor<5>, options: MaxPoolOptions<3>) -> Tensor<5> {
    let kernel_size = options.kernel_size;
    let stride = options.stride;
    let dilation = options.dilation;
    assert!(
        kernel_size.iter().all(|&k| k > 0),
        "max_pool3d: kernel_size must be > 0, got {kernel_size:?}"
    );
    assert!(
        stride.iter().all(|&s| s > 0),
        "max_pool3d: stride must be > 0, got {stride:?}"
    );
    assert!(
        dilation.iter().all(|&d| d > 0),
        "max_pool3d: dilation must be > 0, got {dilation:?}"
    );
    let dims = x.dims();
    let (x, padding) = pad_max_pool_input(x, &options);
    for i in 0..3 {
        assert!(
            padding[i] <= kernel_size[i] / 2,
            "max_pool3d: padding must be <= kernel_size / 2, got padding={:?}, kernel_size={:?}",
            padding,
            kernel_size
        );
    }
    let output = Tensor::new(BridgeTensor::float(Dispatch::max_pool3d(
        x.primitive.into_float(),
        kernel_size,
        stride,
        padding,
        dilation,
        options.ceil_mode,
    )));

    drop_end_padding_windows(output, dims, stride, options.padding, options.ceil_mode)
}

/// Applies a [2D avg pooling](burn_backend::ops::ModuleOps::avg_pool2d).
///
/// Supports symmetric and asymmetric padding through [`AvgPoolOptions`].
pub fn avg_pool2d(x: Tensor<4>, options: AvgPoolOptions<2>) -> Tensor<4> {
    avg_pool(x, &options, |x, padding, count_include_pad| {
        Tensor::new(BridgeTensor::float(Dispatch::avg_pool2d(
            x.primitive.into_float(),
            options.kernel_size,
            options.stride,
            padding,
            count_include_pad,
            options.ceil_mode,
        )))
    })
}

/// Applies a [3D avg pooling](burn_backend::ops::ModuleOps::avg_pool3d).
///
/// Supports symmetric and asymmetric padding through [`AvgPoolOptions`].
///
/// # Panics
///
/// - If any dimension of `kernel_size` is 0.
/// - If any dimension of `stride` is 0.
/// - If any dimension of `padding` exceeds `kernel_size / 2`.
pub fn avg_pool3d(x: Tensor<5>, options: AvgPoolOptions<3>) -> Tensor<5> {
    let kernel_size = options.kernel_size;
    let stride = options.stride;
    assert!(
        kernel_size.iter().all(|&k| k > 0),
        "avg_pool3d: kernel_size must be > 0, got {kernel_size:?}"
    );
    assert!(
        stride.iter().all(|&s| s > 0),
        "avg_pool3d: stride must be > 0, got {stride:?}"
    );
    avg_pool(x, &options, |x, padding, count_include_pad| {
        for i in 0..3 {
            assert!(
                padding[i] <= kernel_size[i] / 2,
                "avg_pool3d: padding must be <= kernel_size / 2, got padding={:?}, kernel_size={:?}",
                padding,
                kernel_size
            );
        }
        Tensor::new(BridgeTensor::float(Dispatch::avg_pool3d(
            x.primitive.into_float(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            options.ceil_mode,
        )))
    })
}

/// Applies a [1D avg pooling](burn_backend::ops::ModuleOps::avg_pool1d).
///
/// Supports symmetric and asymmetric padding through [`AvgPoolOptions`].
pub fn avg_pool1d(x: Tensor<3>, options: AvgPoolOptions<1>) -> Tensor<3> {
    avg_pool(x, &options, |x, padding, count_include_pad| {
        Tensor::new(BridgeTensor::float(Dispatch::avg_pool1d(
            x.primitive.into_float(),
            options.kernel_size[0],
            options.stride[0],
            padding[0],
            count_include_pad,
            options.ceil_mode,
        )))
    })
}

/// Applies a [1D max pooling with indices](burn_backend::ops::ModuleOps::max_pool1d_with_indices).
///
/// Supports symmetric and asymmetric padding through [`MaxPoolOptions`].
/// Returned indices always refer to positions in the unpadded input.
pub fn max_pool1d_with_indices(
    x: Tensor<3>,
    options: MaxPoolOptions<1>,
) -> (Tensor<3>, Tensor<3, Int>) {
    let dims = x.dims();
    let [_, _, length] = dims;
    let indices_dtype = x.device().get_or_init_settings().int_dtype;
    let (x, padding) = pad_max_pool_input(x, &options);
    let output = Dispatch::max_pool1d_with_indices(
        x.primitive.into_float(),
        options.kernel_size[0],
        options.stride[0],
        padding[0],
        options.dilation[0],
        options.ceil_mode,
        indices_dtype,
    );
    let mut indices = Tensor::<3, Int>::new(BridgeTensor::int(output.indices));

    if options.is_asymmetric() {
        let (left, _) = options.padding[0];
        indices = unpad_indices(indices, left, length);
    }

    let output = Tensor::new(BridgeTensor::float(output.output));
    let MaxPoolOptions {
        stride,
        padding,
        ceil_mode,
        ..
    } = options;
    (
        drop_end_padding_windows(output, dims, stride, padding, ceil_mode),
        drop_end_padding_windows(indices, dims, stride, padding, ceil_mode),
    )
}

/// Applies a [2D max pooling with indices](burn_backend::ops::ModuleOps::max_pool2d_with_indices).
///
/// Supports symmetric and asymmetric padding through [`MaxPoolOptions`].
/// Returned indices always refer to positions in the unpadded input.
pub fn max_pool2d_with_indices(
    x: Tensor<4>,
    options: MaxPoolOptions<2>,
) -> (Tensor<4>, Tensor<4, Int>) {
    let dims = x.dims();
    let [_, _, height, width] = dims;
    let indices_dtype = x.device().get_or_init_settings().int_dtype;
    let (x, padding) = pad_max_pool_input(x, &options);
    let output = Dispatch::max_pool2d_with_indices(
        x.primitive.into_float(),
        options.kernel_size,
        options.stride,
        padding,
        options.dilation,
        options.ceil_mode,
        indices_dtype,
    );
    let mut indices = Tensor::<4, Int>::new(BridgeTensor::int(output.indices));

    if options.is_asymmetric() {
        let [(top, _), (left, right)] = options.padding;
        let width_padded = width + left + right;
        let rows = unpad_indices(indices.clone().div_scalar(width_padded as i64), top, height);
        let cols = unpad_indices(indices.remainder_scalar(width_padded as i64), left, width);
        indices = rows.mul_scalar(width as i64).add(cols);
    }

    let output = Tensor::new(BridgeTensor::float(output.output));
    let MaxPoolOptions {
        stride,
        padding,
        ceil_mode,
        ..
    } = options;
    (
        drop_end_padding_windows(output, dims, stride, padding, ceil_mode),
        drop_end_padding_windows(indices, dims, stride, padding, ceil_mode),
    )
}

/// When any dimension has asymmetric padding, materializes all padding with `-inf`
/// so the backend pools without padding.
///
/// Returns the input unchanged along with the backend padding when it is symmetric.
fn pad_max_pool_input<const D: usize, const N: usize>(
    x: Tensor<D>,
    options: &MaxPoolOptions<N>,
) -> (Tensor<D>, [usize; N]) {
    if options.is_asymmetric() {
        (
            x.pad(options.padding, PadMode::Constant(f32::NEG_INFINITY)),
            [0; N],
        )
    } else {
        (x, options.padding.map(|(begin, _)| begin))
    }
}

/// Maps positions along one padded dimension back to the unpadded input.
///
/// A window made only of padding and `-inf` inputs may select a padded position,
/// which is clamped to the nearest input position.
fn unpad_indices<const D: usize>(
    indices: Tensor<D, Int>,
    begin: usize,
    size: usize,
) -> Tensor<D, Int> {
    indices.sub_scalar(begin as i64).clamp(0, size as i64 - 1)
}

/// Average pooling with asymmetric padding support.
///
/// `pool` runs the backend operation with the given symmetric padding and
/// `count_include_pad` flag.
fn avg_pool<const D: usize, const N: usize>(
    x: Tensor<D>,
    options: &AvgPoolOptions<N>,
    pool: impl Fn(Tensor<D>, [usize; N], bool) -> Tensor<D>,
) -> Tensor<D> {
    if !options.is_asymmetric() {
        let padding = options.padding.map(|(begin, _)| begin);
        return pool(x, padding, options.count_include_pad);
    }

    let dims = x.dims();
    let valid = (!options.count_include_pad).then(|| {
        // Only the spatial dimensions matter for the validity mask.
        let mut shape = [1; D];
        shape[D - N..].copy_from_slice(&dims[D - N..]);
        Tensor::<D>::ones(shape, (&x.device(), x.dtype()))
            .pad(options.padding, PadMode::Constant(0.0))
    });
    let output = pool(
        x.pad(options.padding, PadMode::Constant(0.0)),
        [0; N],
        options.count_include_pad,
    );

    let output = match valid {
        // Materialized padding is indistinguishable from input to the backend. Pooling a
        // validity mask with the same settings recovers the fraction of real values in
        // each window, including partial windows created by ceil mode.
        Some(valid) => {
            let valid = pool(valid, [0; N], false);
            let empty = valid.clone().equal_elem(0.0);
            output / valid.mask_fill(empty, 1.0)
        }
        None => output,
    };

    drop_end_padding_windows(
        output,
        dims,
        options.stride,
        options.padding,
        options.ceil_mode,
    )
}

/// In ceil mode, drops the last window of each spatial dimension when it starts in the
/// end padding, matching PyTorch and ONNX.
///
/// Backends only drop such windows for padding they apply themselves, so one extra window
/// survives once asymmetric padding has been materialized. For backend-applied padding
/// the condition never holds and the output is returned unchanged.
///
/// One window is all there is to drop when the end padding is smaller than the kernel (the
/// range PyTorch and ONNX Runtime accept): floor mode never starts a window in it, and ceil
/// mode adds at most one. Larger end padding keeps floor-mode windows there, and only the
/// extra ceil window is dropped, as in the PyTorch and ONNX output-size formulas.
fn drop_end_padding_windows<const D: usize, const N: usize, K: Basic>(
    output: Tensor<D, K>,
    input_dims: [usize; D],
    stride: [usize; N],
    padding: [(usize, usize); N],
    ceil_mode: bool,
) -> Tensor<D, K> {
    if !ceil_mode {
        return output;
    }

    (0..N).fold(output, |output, i| {
        let dim = D - N + i;
        let size = output.dims()[dim];
        let (begin, _) = padding[i];
        if size > 1 && (size - 1) * stride[i] >= input_dims[dim] + begin {
            output.narrow(dim, 0, size - 1)
        } else {
            output
        }
    })
}

/// Applies a [3D max pooling with indices](burn_backend::ops::ModuleOps::max_pool3d_with_indices).
///
/// Supports symmetric and asymmetric padding through [`MaxPoolOptions`].
/// Returned indices always refer to positions in the unpadded input.
///
/// # Panics
///
/// - If any dimension of `kernel_size` is 0.
/// - If any dimension of `stride` is 0.
/// - If any dimension of `dilation` is 0.
/// - If any dimension of `padding` exceeds `kernel_size / 2`.
pub fn max_pool3d_with_indices(
    x: Tensor<5>,
    options: MaxPoolOptions<3>,
) -> (Tensor<5>, Tensor<5, Int>) {
    let kernel_size = options.kernel_size;
    let stride = options.stride;
    let dilation = options.dilation;
    assert!(
        kernel_size.iter().all(|&k| k > 0),
        "max_pool3d_with_indices: kernel_size must be > 0, got {kernel_size:?}"
    );
    assert!(
        stride.iter().all(|&s| s > 0),
        "max_pool3d_with_indices: stride must be > 0, got {stride:?}"
    );
    assert!(
        dilation.iter().all(|&d| d > 0),
        "max_pool3d_with_indices: dilation must be > 0, got {dilation:?}"
    );
    let dims = x.dims();
    let [_, _, depth, height, width] = dims;
    let indices_dtype = x.device().get_or_init_settings().int_dtype;
    let (x, padding) = pad_max_pool_input(x, &options);
    for i in 0..3 {
        assert!(
            padding[i] <= kernel_size[i] / 2,
            "max_pool3d_with_indices: padding must be <= kernel_size / 2, got padding={:?}, kernel_size={:?}",
            padding,
            kernel_size
        );
    }
    let output = Dispatch::max_pool3d_with_indices(
        x.primitive.into_float(),
        kernel_size,
        stride,
        padding,
        dilation,
        options.ceil_mode,
        indices_dtype,
    );
    let mut indices = Tensor::<5, Int>::new(BridgeTensor::int(output.indices));

    if options.is_asymmetric() {
        let [(front, _), (top, bottom), (left, right)] = options.padding;
        let width_padded = width + left + right;
        let height_padded = height + top + bottom;
        let spatial_slice_padded = height_padded * width_padded;
        let depth_indices = unpad_indices(
            indices.clone().div_scalar(spatial_slice_padded as i64),
            front,
            depth,
        );
        let rem = indices.remainder_scalar(spatial_slice_padded as i64);
        let rows = unpad_indices(rem.clone().div_scalar(width_padded as i64), top, height);
        let cols = unpad_indices(rem.remainder_scalar(width_padded as i64), left, width);
        indices = depth_indices
            .mul_scalar((height * width) as i64)
            .add(rows.mul_scalar(width as i64))
            .add(cols);
    }

    let output = Tensor::new(BridgeTensor::float(output.output));
    let MaxPoolOptions {
        stride,
        padding,
        ceil_mode,
        ..
    } = options;
    (
        drop_end_padding_windows(output, dims, stride, padding, ceil_mode),
        drop_end_padding_windows(indices, dims, stride, padding, ceil_mode),
    )
}
/// Applies a [2D adaptive avg pooling](burn_backend::ops::ModuleOps::adaptive_avg_pool2d).
pub fn adaptive_avg_pool2d(x: Tensor<4>, output_size: [usize; 2]) -> Tensor<4> {
    Tensor::new(BridgeTensor::float(Dispatch::adaptive_avg_pool2d(
        x.primitive.into_float(),
        output_size,
    )))
}

/// Applies a [3D adaptive avg pooling](burn_backend::ops::ModuleOps::adaptive_avg_pool3d).
pub fn adaptive_avg_pool3d(x: Tensor<5>, output_size: [usize; 3]) -> Tensor<5> {
    Tensor::new(BridgeTensor::float(Dispatch::adaptive_avg_pool3d(
        x.primitive.into_float(),
        output_size,
    )))
}

/// Applies a [1D adaptive avg pooling](burn_backend::ops::ModuleOps::adaptive_avg_pool1d).
pub fn adaptive_avg_pool1d(x: Tensor<3>, output_size: usize) -> Tensor<3> {
    Tensor::new(BridgeTensor::float(Dispatch::adaptive_avg_pool1d(
        x.primitive.into_float(),
        output_size,
    )))
}

/// Applies a [2D interpolation](burn_backend::ops::ModuleOps::interpolate).
///
/// The output spatial size is taken from `options.output_size`, or computed as
/// `floor(input_size * scale_factor)` from `options.scale_factor`.
///
/// # Panics
///
/// Panics unless exactly one of `output_size` or `scale_factor` is set, or if the
/// scaled size exceeds `usize::MAX`.
///
/// # Example
///
/// ```rust,ignore
/// // Resize to a fixed size.
/// interpolate(x, InterpolateOptions::new(mode).with_output_size([224, 224]));
/// // Upsample by 2x.
/// interpolate(x, InterpolateOptions::new(mode).with_scale_factor([2.0, 2.0]));
/// ```
pub fn interpolate(x: Tensor<4>, options: InterpolateOptions) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    let output_size = interpolate_output_size([h, w], &options);
    Tensor::new(BridgeTensor::float(Dispatch::interpolate(
        x.primitive.into_float(),
        output_size,
        options,
    )))
}

fn interpolate_output_size(input_size: [usize; 2], options: &InterpolateOptions) -> [usize; 2] {
    match (options.output_size, options.scale_factor) {
        (Some(output_size), None) => output_size,
        (None, Some(scale_factor)) => core::array::from_fn(|i| {
            let size = input_size[i] as f64 * scale_factor[i] as f64;
            assert!(
                size <= usize::MAX as f64,
                "Interpolate scale factor {} is too large for input size {}",
                scale_factor[i],
                input_size[i]
            );
            size as usize
        }),
        (Some(_), Some(_)) => {
            panic!("Interpolate options must set only one of output_size or scale_factor")
        }
        (None, None) => panic!("Interpolate options must set output_size or scale_factor"),
    }
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

    // A quantized weight must stay quantized: `linear_impl` converts its
    // operands to float, which would dequantize (materialize) the whole weight
    // matrix on every forward. Route through the quantized matmul instead, which
    // streams the packed weight directly — but reuse the same batch-fold policy
    // the float `linear` applies, so a decode-shaped call folds its batches into
    // the rows for one `[rows, d_in] @ [d_in, d_out]` matmul rather than a
    // broadcast batched matmul that re-reads the packed weight per batch.
    if let DType::QFloat(_) = weight.dtype() {
        let dims = input.dims();
        let analysis = MatmulTransformAnalysis::from_shapes(&input.shape(), &weight.shape());

        let output = match MatmulTransformPolicy::default().action(&analysis) {
            MatmulTransformAction::MergeBatches { rows } => {
                let d_in = dims[D - 1];
                let d_out = weight.dims()[1];

                let folded = input.reshape([rows, d_in]).matmul(weight);

                let mut out_dims = dims;
                out_dims[D - 1] = d_out;
                folded.reshape(out_dims)
            }
            MatmulTransformAction::Keep => input.matmul(weight.unsqueeze::<D>()),
        };

        return match bias {
            Some(bias) => output + bias.unsqueeze(),
            None => output,
        };
    }

    Tensor::new(linear_impl(
        input.primitive,
        weight.primitive,
        bias.map(|b| b.primitive),
    ))
}

fn linear_impl(
    input: BridgeTensor,
    weight: BridgeTensor,
    bias: Option<BridgeTensor>,
) -> BridgeTensor {
    BridgeTensor::float(Dispatch::linear(
        input.into_float(),
        weight.into_float(),
        bias.map(|b| b.into_float()),
    ))
}

/// Computes scaled dot-product attention: softmax(QKᵗ * scale) · V,
/// where scale defaults to 1/sqrt(head_dim) (configurable via `options.scale`).
/// Optionally applies masking, additive bias, causal masking, and softcap.
///
/// Scores are computed as `softcap(QKᵗ · scale)`, then masked (`mask` and causal) to
/// `-inf`, then `attn_bias` is added, then softmax is taken; see
/// [`ModuleOps::attention`](burn_backend::ops::ModuleOps::attention) for the full contract.
///
/// # Arguments
/// - `query`: Query tensor of shape `[batch_size, num_heads, seq_len_q, head_dim]`
/// - `key`: Key tensor of shape `[batch_size, num_kv_heads, seq_len_k, head_dim]`
/// - `value`: Value tensor of shape `[batch_size, num_kv_heads, seq_len_k, val_dim]`
///
///   `num_heads` must be a multiple of `num_kv_heads` (grouped-query attention). Query
///   head `h` attends with K/V head `h / (num_heads / num_kv_heads)`.
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
    burn_backend::ops::attention::AttentionShapes::new(
        &query.shape(),
        &key.shape(),
        &value.shape(),
    );
    Tensor::new(BridgeTensor::float(Dispatch::attention(
        query.primitive.into_float(),
        key.primitive.into_float(),
        value.primitive.into_float(),
        mask.map(|mask| mask.primitive.into()),
        attn_bias.map(|bias| bias.primitive.into_float()),
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
    Tensor::new(BridgeTensor::float(
        burn_backend::ops::attention::attention_fallback::<Dispatch>(
            query.primitive.into_float(),
            key.primitive.into_float(),
            value.primitive.into_float(),
            mask.map(|mask| mask.primitive.into()),
            attn_bias.map(|bias| bias.primitive.into_float()),
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
    Tensor::new(BridgeTensor::float(Dispatch::conv2d_weight_backward(
        x.primitive.into_float(),
        weight.primitive.into_float(),
        output_grad.primitive.into_float(),
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
    Tensor::new(BridgeTensor::float(Dispatch::avg_pool2d_backward(
        x.primitive.into_float(),
        grad.primitive.into_float(),
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
    Tensor::new(BridgeTensor::float(
        Dispatch::max_pool2d_with_indices_backward(
            x.primitive.into_float(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad.primitive.into_float(),
            indices.primitive.into(),
        )
        .x_grad,
    ))
}

/// Backward pass for the [avg pooling 3d](ModuleOps::avg_pool3d) operation.
///
/// # Panics
///
/// - If any dimension of `kernel_size` is 0.
/// - If any dimension of `stride` is 0.
/// - If any dimension of `padding` exceeds `kernel_size / 2`.
/// - If `grad` dimensions do not match the expected forward output shape.
pub fn avg_pool3d_backward(
    x: Tensor<5>,
    grad: Tensor<5>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    count_include_pad: bool,
    ceil_mode: bool,
) -> Tensor<5> {
    assert!(
        kernel_size.iter().all(|&k| k > 0),
        "avg_pool3d_backward: kernel_size must be > 0, got {kernel_size:?}"
    );
    assert!(
        stride.iter().all(|&s| s > 0),
        "avg_pool3d_backward: stride must be > 0, got {stride:?}"
    );
    for i in 0..3 {
        assert!(
            padding[i] <= kernel_size[i] / 2,
            "avg_pool3d_backward: padding must be <= kernel_size / 2, got padding={:?}, kernel_size={:?}",
            padding,
            kernel_size
        );
    }
    let [batch_size, channels, d_in, h_in, w_in] = x.dims();
    let expected_output_dims = [
        batch_size,
        channels,
        burn_backend::ops::conv::calculate_pool_output_size(
            kernel_size[0],
            stride[0],
            padding[0],
            1,
            d_in,
            ceil_mode,
        ),
        burn_backend::ops::conv::calculate_pool_output_size(
            kernel_size[1],
            stride[1],
            padding[1],
            1,
            h_in,
            ceil_mode,
        ),
        burn_backend::ops::conv::calculate_pool_output_size(
            kernel_size[2],
            stride[2],
            padding[2],
            1,
            w_in,
            ceil_mode,
        ),
    ];
    assert_eq!(
        grad.dims(),
        expected_output_dims,
        "grad shape {:?} must match expected forward output shape {:?}",
        grad.dims(),
        expected_output_dims
    );
    Tensor::new(BridgeTensor::float(Dispatch::avg_pool3d_backward(
        x.primitive.into_float(),
        grad.primitive.into_float(),
        kernel_size,
        stride,
        padding,
        count_include_pad,
        ceil_mode,
    )))
}

/// Backward pass for the [max pooling 3d](ModuleOps::max_pool3d_with_indices) operation.
///
/// # Panics
///
/// - If any dimension of `kernel_size` is 0.
/// - If any dimension of `stride` is 0.
/// - If any dimension of `dilation` is 0.
/// - If any dimension of `padding` exceeds `kernel_size / 2`.
/// - If `indices` and `output_grad` shapes do not match.
#[allow(clippy::too_many_arguments)]
pub fn max_pool3d_with_indices_backward(
    x: Tensor<5>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
    output_grad: Tensor<5>,
    indices: Tensor<5, Int>,
) -> Tensor<5> {
    assert!(
        kernel_size.iter().all(|&k| k > 0),
        "max_pool3d_with_indices_backward: kernel_size must be > 0, got {kernel_size:?}"
    );
    assert!(
        stride.iter().all(|&s| s > 0),
        "max_pool3d_with_indices_backward: stride must be > 0, got {stride:?}"
    );
    assert!(
        dilation.iter().all(|&d| d > 0),
        "max_pool3d_with_indices_backward: dilation must be > 0, got {dilation:?}"
    );
    for i in 0..3 {
        assert!(
            padding[i] <= kernel_size[i] / 2,
            "max_pool3d_with_indices_backward: padding must be <= kernel_size / 2, got padding={:?}, kernel_size={:?}",
            padding,
            kernel_size
        );
    }
    assert_eq!(
        indices.dims(),
        output_grad.dims(),
        "max_pool3d_with_indices_backward: indices and output_grad must have the same dimensions, got {:?} and {:?}",
        indices.dims(),
        output_grad.dims()
    );
    Tensor::new(BridgeTensor::float(
        Dispatch::max_pool3d_with_indices_backward(
            x.primitive.into_float(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad.primitive.into_float(),
            indices.primitive.into(),
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
    Tensor::new(layer_norm_impl(
        input.primitive,
        gamma.primitive,
        beta.map(|b| b.primitive),
        epsilon,
    ))
}

fn layer_norm_impl(
    input: BridgeTensor,
    gamma: BridgeTensor,
    beta: Option<BridgeTensor>,
    epsilon: f64,
) -> BridgeTensor {
    BridgeTensor::float(Dispatch::layer_norm(
        input.into_float(),
        gamma.into_float(),
        beta.map(|b| b.into_float()),
        epsilon,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::InterpolateMode;

    fn options() -> InterpolateOptions {
        InterpolateOptions::new(InterpolateMode::Nearest)
    }

    #[test]
    fn interpolate_output_size_from_output_size() {
        let size = interpolate_output_size([4, 4], &options().with_output_size([2, 3]));
        assert_eq!(size, [2, 3]);
    }

    #[test]
    fn interpolate_output_size_from_scale_factor_floors() {
        let size = interpolate_output_size([4, 5], &options().with_scale_factor([2.0, 1.5]));
        assert_eq!(size, [8, 7]);
    }

    #[test]
    fn interpolate_options_last_sizing_builder_wins() {
        let size = interpolate_output_size(
            [4, 4],
            &options()
                .with_output_size([2, 2])
                .with_scale_factor([2.0, 2.0]),
        );
        assert_eq!(size, [8, 8]);
    }

    #[test]
    #[should_panic(expected = "must set output_size or scale_factor")]
    fn interpolate_output_size_requires_size() {
        interpolate_output_size([4, 4], &options());
    }

    #[test]
    #[should_panic(expected = "only one of output_size or scale_factor")]
    fn interpolate_output_size_rejects_both() {
        let mut options = options().with_output_size([2, 2]);
        options.scale_factor = Some([2.0, 2.0]);
        interpolate_output_size([4, 4], &options);
    }

    #[test]
    #[should_panic(expected = "too large")]
    fn interpolate_output_size_rejects_overflow() {
        interpolate_output_size(
            [4, usize::MAX - 1],
            &options().with_scale_factor([1.0, 2.0]),
        );
    }

    #[test]
    #[should_panic(expected = "kernel_size must be > 0")]
    fn test_max_pool3d_kernel_size_zero_panics() {
        let tensor = Tensor::<5>::zeros([1, 1, 4, 4, 4], &Default::default());
        let options = MaxPoolOptions {
            kernel_size: [0, 2, 2],
            stride: [1, 1, 1],
            padding: [(0, 0); 3],
            dilation: [1, 1, 1],
            ceil_mode: false,
        };
        max_pool3d(tensor, options);
    }

    #[test]
    #[should_panic(expected = "stride must be > 0")]
    fn test_max_pool3d_stride_zero_panics() {
        let tensor = Tensor::<5>::zeros([1, 1, 4, 4, 4], &Default::default());
        let options = MaxPoolOptions {
            kernel_size: [2, 2, 2],
            stride: [0, 1, 1],
            padding: [(0, 0); 3],
            dilation: [1, 1, 1],
            ceil_mode: false,
        };
        max_pool3d(tensor, options);
    }

    #[test]
    #[should_panic(expected = "dilation must be > 0")]
    fn test_max_pool3d_dilation_zero_panics() {
        let tensor = Tensor::<5>::zeros([1, 1, 4, 4, 4], &Default::default());
        let options = MaxPoolOptions {
            kernel_size: [2, 2, 2],
            stride: [1, 1, 1],
            padding: [(0, 0); 3],
            dilation: [0, 1, 1],
            ceil_mode: false,
        };
        max_pool3d(tensor, options);
    }

    #[test]
    #[should_panic(expected = "max_pool3d: padding must be <= kernel_size / 2")]
    fn test_max_pool3d_padding_greater_than_half_kernel_panics() {
        let tensor = Tensor::<5>::zeros([1, 1, 4, 4, 4], &Default::default());
        let options = MaxPoolOptions::new([2, 2, 2]).with_padding([2, 0, 0]);
        max_pool3d(tensor, options);
    }

    #[test]
    #[should_panic(expected = "grad shape")]
    fn test_avg_pool3d_backward_grad_shape_mismatch_panics() {
        let tensor = Tensor::<5>::zeros([1, 1, 4, 4, 4], &Default::default());
        let grad = Tensor::<5>::zeros([1, 1, 2, 2, 2], &Default::default());
        avg_pool3d_backward(tensor, grad, [2, 2, 2], [1, 1, 1], [0, 0, 0], true, false);
    }
}
