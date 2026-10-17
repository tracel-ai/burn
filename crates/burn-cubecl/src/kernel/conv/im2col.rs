use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{
    DType,
    ops::{ConvOptions, conv::calculate_conv_output_sizes},
};
use burn_std::{Metadata, Shape};
use core::iter;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{
        FastDivmod,
        tensor::{TensorHandle, into_contiguous_pitched, layout::linear::LinearViewMut},
    },
};
use cubek::convolution::components::ConvSetupError;

use crate::{
    CubeDevice,
    kernel::{
        AddOp, into_contiguous_aligned, launch_binop,
        matmul::{MatmulStrategy, matmul},
        reduce::{KernelReduceStrategy, reduce_dim},
        utils::{address_type, decompose_linear, shape_divmod, split_dim},
    },
    ops::{numeric::empty_device_dtype, reshape, swap_dims},
    tensor::CubeTensor,
};
use cubek::reduce::components::instructions::ReduceOperationConfig;

#[cfg(not(test))]
pub(crate) fn batches_per_run(
    batch_size: usize,
    out_shape: usize,
    plane_size: usize,
) -> Result<usize, ConvSetupError> {
    use cubek::matmul::definition::MatmulAvailabilityError;

    let cube_count_per_batch = out_shape.div_ceil(plane_size);
    let max_cube_count = u16::MAX as usize;
    let max_simultaneous = Ord::min(max_cube_count / cube_count_per_batch, batch_size);
    if max_simultaneous == 0 {
        return Err(MatmulAvailabilityError::CubeCountTooBig(CubeCount::Static(
            cube_count_per_batch as u32,
            1,
            1,
        ))
        .into());
    }
    Ok((0..=max_simultaneous)
        .rev()
        .find(|per_run| batch_size.is_multiple_of(*per_run))
        .expect("Logically not possible"))
}

#[cfg(test)]
#[allow(unused)]
pub(crate) fn batches_per_run(
    batch_size: usize,
    out_shape: usize,
    plane_size: usize,
) -> Result<usize, ConvSetupError> {
    Ok(1)
}

pub fn conv_im2col_1x1<const N: usize>(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    let rank = input.meta.num_dims();
    let dim_c = rank - 1;

    let out_channels = weight.meta.shape()[0];

    check_pointwise_strided(&weight.meta.shape()[1..dim_c], &options)?;

    let out_shape = calculate_conv_output_sizes(
        &weight.meta.shape()[1..dim_c],
        &options.stride,
        &options.padding,
        &options.dilation,
        &input.meta.shape()[1..dim_c],
    );

    let mut split_m = vec![input.meta.shape()[0]];
    split_m.extend(out_shape.iter().copied());

    let input = match options.stride.iter().all(|stride| *stride == 1) {
        true => input,
        false => strided_spatial_view(input, &out_shape, &options.stride),
    };

    let input = reshape_input(input); // [(NHW), C] : [M, K]
    let dtype = input.dtype;

    // Permute to N-major, while keeping memory layout K-major. K-major for both sides is the most
    // efficient for matmul, and allows skipping a contiguous kernel
    let weight = swap_dims(reshape_weight(weight), 0, 1); // [K, N]

    let out = matmul(input, weight, None, MatmulStrategy::default(), dtype)?; // [M, N]

    // Skip reshape to avoid potential `into_contiguous`. We're only splitting dims so it's safe.
    let mut out = split_dim(out, 0, &split_m); // [N, H, W, C]

    if let Some(bias) = bias {
        let mut bias_shape = iter::repeat_n(1, rank - 1).collect::<Vec<_>>();
        bias_shape.push(out_channels);
        let bias = reshape(bias, bias_shape.into());
        out = launch_binop::<AddOp>(out, bias);
    }

    Ok(out)
}

/// Reshapes NHWC input to [(N, H, W), C]
fn reshape_input(input: CubeTensor) -> CubeTensor {
    let input = crate::kernel::untile(input);
    let rank = input.meta.num_dims();
    let dim_c = rank - 1;
    let dtype = input.dtype;

    let batch_size = input.meta.shape()[0];
    let in_c: usize = input.meta.shape()[dim_c];
    let in_shape: Shape = input.meta.shape()[1..dim_c].into();

    let mut input = if !is_spatial_contiguous(input.meta.shape(), input.meta.strides()) {
        let (client, device) = (input.client.clone(), input.device.clone());
        let contiguous =
            into_contiguous_pitched(&client, input.binding(), dtype_to_storage_type(dtype));
        from_handle(client, device, contiguous, dtype)
    } else {
        input
    };

    *input.meta = Metadata::new(
        [batch_size * in_shape.num_elements(), in_c], // [M, K]
        [input.meta.strides()[dim_c - 1], input.meta.strides()[dim_c]],
    );
    input
}

fn is_spatial_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    let rank = shape.len();
    let dim_c = rank - 1;

    // Channel must be contiguous for the [(N, H, W), C] reshape to be valid
    if strides[dim_c] != 1 {
        return false;
    }

    for i in (1..dim_c).rev() {
        if strides[i + 1] * shape[i + 1] != strides[i] {
            return false;
        }
    }
    true
}

fn from_handle(
    client: Client,
    device: CubeDevice,
    handle: TensorHandle,
    dtype: DType,
) -> CubeTensor {
    CubeTensor::new(
        client.clone(),
        handle.handle,
        *handle.metadata,
        device.clone(),
        dtype,
    )
}

/// Errors unless the convolution is pointwise, the case this module reduces to
/// a single matmul.
///
/// A 1x1 convolution with unit stride, no padding and no dilation maps every
/// output pixel to the input pixel under it, so `im2col` is the identity and
/// the convolution is a per-pixel `[C_in, C_out]` matmul. A 1x1 that strides or
/// pads reads outside its own pixel and is declined, even where its output
/// happens to come back the size of its input — `in = 2 * padding + 1` under a
/// stride of 2 is such a shape.
///
/// The shapes are NHWC, as everything below `conv/base.rs` is.
fn check_pointwise<const N: usize>(
    kernel_shape: &[usize],
    options: &ConvOptions<N>,
) -> Result<(), ConvSetupError> {
    check_pointwise_strided(kernel_shape, options)?;

    match options.stride.iter().all(|stride| *stride == 1) {
        true => Ok(()),
        false => Err(ConvSetupError::Unknown),
    }
}

/// The same, minus the stride: a strided pointwise convolution reads a regular
/// subset of its input, which a view can hold exactly, so the forward pass
/// still lowers to one matmul. The gradient paths have no such view and keep
/// [`check_pointwise`].
fn check_pointwise_strided<const N: usize>(
    kernel_shape: &[usize],
    options: &ConvOptions<N>,
) -> Result<(), ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let pointwise = kernel_shape.iter().all(|size| *size == 1)
        && options
            .padding
            .iter()
            .all(|&(begin, end)| begin == 0 && end == 0)
        && options.dilation.iter().all(|dilation| *dilation == 1);

    match pointwise {
        true => Ok(()),
        false => Err(ConvSetupError::Unknown),
    }
}

/// The view a strided pointwise convolution actually reads: every `stride`-th
/// position along each spatial dim. Multiplying the spatial strides leaves the
/// contiguous copy in [`reshape_input`] to gather exactly those rows.
fn strided_spatial_view(input: CubeTensor, out_shape: &[usize], stride: &[usize]) -> CubeTensor {
    let mut input = crate::kernel::untile(input);
    let mut shape = input.meta.shape().to_vec();
    let mut strides = input.meta.strides().to_vec();

    for (dim, (out, step)) in out_shape.iter().zip(stride).enumerate() {
        shape[dim + 1] = *out;
        strides[dim + 1] *= *step;
    }

    *input.meta = Metadata::new(shape, strides);
    input
}

/// Drops a pointwise weight's unit kernel dimensions, giving `[C_out, C_in]`.
///
/// Rewriting the metadata rather than reshaping keeps a padded channel stride,
/// so a weight the pitched allocator already aligned for TMA is not copied to
/// say so. One that is not gets a pitched copy here rather than a second kernel
/// inside the matmul.
fn reshape_weight(weight: CubeTensor) -> CubeTensor {
    let mut weight = crate::kernel::untile(weight);
    let dim_c = weight.meta.num_dims() - 1;
    let strides = [weight.meta.strides()[0], weight.meta.strides()[dim_c]];
    let shape = [weight.meta.shape()[0], weight.meta.shape()[dim_c]];

    *weight.meta = Metadata::new(shape, strides);

    match strides[1] {
        1 => weight,
        _ => into_contiguous_aligned(weight),
    }
}

/// Data gradient of a dense convolution: `[M, C_out] @ [C_out, taps * C_in]`,
/// then [`col2im`].
///
/// The matmul yields the gradient of the im2col columns `[M, taps * C_in]`;
/// [`col2im`] scatter-adds them onto the input gradient — each input pixel
/// receives the sum over every tap whose window read it. Declines `groups != 1`
/// and pointwise unit stride (kept for [`dgrad_im2col_1x1`]'s unmaterialised
/// matmul); anchored key fields are never declined explicitly.
pub fn dgrad_im2col<const N: usize>(
    out_grad: CubeTensor,
    weight: CubeTensor,
    input_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    let dim_c = out_grad.meta.num_dims() - 1;
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }
    if check_pointwise(&weight.meta.shape()[1..dim_c], &options).is_ok() {
        return Err(ConvSetupError::Unknown);
    }

    let kernel_shape = weight.meta.shape()[1..dim_c].to_vec();
    let out_channels = weight.meta.shape()[0];
    let in_channels = weight.meta.shape()[dim_c];
    let taps = kernel_shape.iter().product::<usize>();
    let out_shape = out_grad.meta.shape()[1..dim_c].to_vec();

    let dtype = out_grad.dtype;
    let out_grad = reshape_input(out_grad);
    let weight = reshape(weight, Shape::new([out_channels, taps * in_channels]));
    let columns = matmul(out_grad, weight, None, MatmulStrategy::default(), dtype)?;
    Ok(col2im::<N>(
        columns,
        &out_shape,
        &input_shape,
        &kernel_shape,
        &options,
    ))
}

/// The gradient of a pointwise convolution with respect to its input, as one
/// matmul.
///
/// `grad_in[(n, h, w), c_in] = sum over c_out of grad_out[(n, h, w), c_out] *
/// weight[c_out, c_in]`. The fallback computes the same thing as a transposed
/// convolution, which has no NHWC path and falls back to a naive kernel on a
/// device with no accelerated matmul for the dtype.
pub fn dgrad_im2col_1x1<const N: usize>(
    out_grad: CubeTensor,
    weight: CubeTensor,
    input_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    let dim_c = out_grad.meta.num_dims() - 1;

    check_pointwise(&weight.meta.shape()[1..dim_c], &options)?;

    let split_m = input_shape[..dim_c].to_vec();

    let out_grad = reshape_input(out_grad); // [(NHW), C_out] : [M, K]
    let dtype = out_grad.dtype;

    // No transpose here, unlike the forward: it wants `[K, N]` and the weight's
    // own order *is* `[C_out, C_in]`, because this reduces over `C_out` where
    // the forward reduces over `C_in`.
    let weight = reshape_weight(weight); // [K, N]

    let out = matmul(out_grad, weight, None, MatmulStrategy::default(), dtype)?; // [M, N]

    // Skip reshape to avoid potential `into_contiguous`. We're only splitting dims so it's safe.
    Ok(split_dim(out, 0, &split_m)) // [N, H, W, C_in]
}

/// The gradient of a pointwise convolution with respect to its weight, as one
/// matmul.
///
/// `grad_w[c_out, c_in] = sum over (n, h, w) of grad_out[(n, h, w), c_out] *
/// input[(n, h, w), c_in]` — a tall reduction into a small output, which the
/// fallback's "convolve the input by the gradient" framing hides from the
/// matmul tuner.
pub fn wgrad_im2col_1x1<const N: usize>(
    input: CubeTensor,
    out_grad: CubeTensor,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    let dim_c = input.meta.num_dims() - 1;

    check_pointwise(&weight_shape[1..dim_c], &options)?;

    let input = reshape_input(input); // [(NHW), C_in] : [M, N]
    let out_grad = reshape_input(out_grad); // [(NHW), C_out] : [M, K]
    let dtype = out_grad.dtype;

    // A metadata swap, so the matmul reads the gradient K-major rather than a
    // transposed copy of it being made.
    let out_grad = swap_dims(out_grad, 0, 1); // [C_out, M]

    let grad = matmul(out_grad, input, None, MatmulStrategy::default(), dtype)?; // [C_out, C_in]

    // Only unit kernel dimensions are being reinserted, so nothing moves.
    Ok(reshape(grad, weight_shape)) // [C_out, 1, .., 1, C_in]
}

/// How few rows of the contraction a single piece may be left with.
///
/// The cut is worth making because it buys parallelism, and stops being worth
/// making once each piece is too short to amortise its own launch. Flat across
/// every piece count [`MAX_SPLIT`] leaves reachable, on the shapes measured, so
/// the exact value is not delicate.
const MIN_SPLIT_ROWS: usize = 2048;

/// The most pieces to cut into, so that the reduction putting them back stays
/// small next to the matmul that produced them.
const MAX_SPLIT: usize = 64;

/// How many pieces to cut a weight gradient's contraction into, or `None` when
/// cutting it does not apply.
///
/// A weight gradient contracts over every pixel in the batch, so `k` is enormous
/// against an output that is only `c_out` by `c_in`. A matmul kernel gives each
/// output element the whole of `k`, which leaves a device with far more lanes
/// than there are output elements mostly idle — the arithmetic is fine and the
/// *shape* is wrong. Cutting `k` into independent pieces multiplies the work
/// items by the cut and costs one small reduction to put back.
///
/// Declines a contraction too short to be worth cutting, and one that none of
/// the cuts it considers divides — the search tries powers of two only, which is
/// where `k = batch * height * width` almost always has its factors, so the rare
/// `k` whose only equal cuts are odd is left uncut rather than sent down a shape
/// nothing measured. Whether the cut *pays* is left to autotune, which measures
/// it against the uncut form on the actual shape — a guess about where the
/// crossover lies would only be a worse version of that measurement.
///
/// The count has to divide `k` exactly, since the whole point is that the
/// reshape splitting it is free.
fn split_count(k: usize) -> Option<usize> {
    if k < MIN_SPLIT_ROWS * 2 {
        return None;
    }

    let by_rows = k / MIN_SPLIT_ROWS;
    let ceiling = Ord::min(by_rows, MAX_SPLIT);

    // Powers of two downward, so the split divides `k` and the pieces are
    // equal. `k` is `batch * height * width` and usually has many factors of
    // two, but nothing guarantees it, so this can come back empty.
    (1..=ceiling.ilog2())
        .rev()
        .map(|log| 1usize << log)
        .find(|split| k.is_multiple_of(*split))
}

/// The gradient with respect to a 1x1 convolution's weight, with the
/// contraction cut into independent pieces and summed.
///
/// Identical arithmetic to [`wgrad_im2col_1x1`] up to the order the products
/// are added in, and the same single matmul underneath — only batched, over a
/// `k` that has been cut. See [`split_count`] for why that is worth doing.
///
/// Registered beside the uncut form rather than replacing it, so that autotune
/// decides per shape: the cut is a large win where the output is small and a
/// small loss where it is not, and which side a shape falls on is exactly the
/// kind of thing measuring answers better than a rule.
pub fn wgrad_im2col_1x1_split<const N: usize>(
    input: CubeTensor,
    out_grad: CubeTensor,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    let dim_c = input.meta.num_dims() - 1;

    check_pointwise(&weight_shape[1..dim_c], &options)?;

    // Every way of bowing out below ends in the uncut form rather than in an
    // `Err`, so that this candidate declines exactly what [`wgrad_im2col_1x1`]
    // declines and nothing more. What it would otherwise decline on turns on
    // `k`, and the autotune key holds the spatial dimensions only anchored: a
    // shape that declines can share a key with one that did not, and the tuner
    // unwraps whatever it already picked, so declining on a cached hit aborts
    // the process.
    let uncut = {
        let args = (
            input.clone(),
            out_grad.clone(),
            weight_shape.clone(),
            options.clone(),
        );
        move || wgrad_im2col_1x1::<N>(args.0, args.1, args.2, args.3)
    };

    let rows: usize = input.meta.shape()[..dim_c].iter().product();
    let Some(split) = split_count(rows) else {
        return uncut();
    };
    let per = rows / split;

    let input = reshape_input(input); // [M, C_in]
    let out_grad = reshape_input(out_grad); // [M, C_out]
    let dtype = out_grad.dtype;

    let in_channels = input.meta.shape()[1];
    let out_channels = out_grad.meta.shape()[1];

    // `[M, C]` -> `[split, M / split, C]`. Free: the contraction is the leading
    // axis of both operands, so cutting it only inserts a dimension.
    let input = reshape(input, Shape::new([split, per, in_channels]));
    let out_grad = reshape(out_grad, Shape::new([split, per, out_channels]));

    // `[split, C_out, M / split] @ [split, M / split, C_in]`, a stride swap on
    // the gradient as in the uncut form.
    let out_grad = swap_dims(out_grad, 1, 2);
    let Ok(partials) = matmul(out_grad, input, None, MatmulStrategy::default(), dtype) else {
        return uncut();
    };

    // `[split, C_out, C_in]` -> `[1, C_out, C_in]`. Small next to the matmul:
    // the pieces are the only thing being added, not the contraction.
    let grad = reduce_dim(
        partials,
        None,
        0,
        KernelReduceStrategy::default(),
        ReduceOperationConfig::Sum,
    );
    // The axis is in range, so only the strategy or the dtype can refuse here —
    // and the uncut form adds the same products inside its own matmul.
    let Ok(grad) = grad else {
        return uncut();
    };

    Ok(reshape(grad, weight_shape))
}

/// Maps `[batch, ...input_spatial, channels]` to
/// `[batch, ...output_spatial, kernel_taps * channels]` in one kernel.
/// Kernel taps use spatial row-major order. Reads outside the input supply zero.
/// Preserves the input dtype and device and supports strided input storage.
/// Returns an error if source coordinates exceed the host index range.
fn im2col<const N: usize>(
    input: CubeTensor,
    out_shape: &[usize],
    kernel_shape: &[usize],
    options: &ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    let input = crate::kernel::untile(input);
    let channels = input.meta.shape()[N + 1];
    let taps = kernel_shape.iter().product::<usize>();
    let mut shape = vec![input.meta.shape()[0]];
    shape.extend_from_slice(out_shape);
    shape.push(taps * channels);
    let mut kernel = SequenceArg::new();
    let mut stride = SequenceArg::new();
    let mut padding = SequenceArg::new();
    let mut dilation = SequenceArg::new();
    let mut coordinate_type = AddressType::U32;
    for (axis, &extent) in kernel_shape.iter().enumerate().take(N) {
        let max_source = out_shape[axis]
            .saturating_sub(1)
            .checked_mul(options.stride[axis])
            .and_then(|output| {
                extent
                    .saturating_sub(1)
                    .checked_mul(options.dilation[axis])
                    .and_then(|tap| output.checked_add(tap))
            })
            .ok_or_else(|| {
                ConvSetupError::InvalidConfig(Box::new("im2col source coordinate exceeds usize"))
            })?;

        // Small buffers can still require wide coordinates or option scalars.
        for value in [
            max_source,
            extent,
            options.stride[axis],
            options.padding[axis].0,
            options.dilation[axis],
        ] {
            coordinate_type = coordinate_type.max(AddressType::from_len(value));
        }
        kernel.push(extent);
        stride.push(options.stride[axis]);
        padding.push(options.padding[axis].0);
        dilation.push(options.dilation[axis]);
    }

    let columns = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape.into(),
        input.dtype,
    );
    let client = input.client.clone();
    let num_elements = columns.meta.num_elements();
    let cube_dim = CubeDim::new(&client, num_elements);
    let cube_count = calculate_cube_count_elemwise(&client, num_elements, cube_dim);

    // Each logical output element has one writer. The kernel checks every source
    // coordinate before reading and uses tensor strides for storage offsets.
    unsafe {
        im2col_kernel::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            address_type!(input, columns).max(coordinate_type),
            input.clone().into_tensor_arg(),
            columns.clone().into_linear_view(),
            shape_divmod(&columns),
            kernel,
            stride,
            padding,
            dilation,
            dtype_to_storage_type(input.dtype),
        );
    }
    Ok(columns)
}

/// Writes columns `[batch, ...output_spatial, kernel_taps * channels]` from
/// an input `[batch, ...input_spatial, channels]`, preserving spatial axis order.
#[cube(launch_unchecked, address_type = "dynamic")]
fn im2col_kernel<E: Numeric>(
    input: &Tensor<E>,
    mut columns: LinearViewMut<'_, E>,
    columns_shape: Sequence<FastDivmod<usize>>,
    kernel: Sequence<usize>,
    stride: Sequence<usize>,
    padding: Sequence<usize>,
    dilation: Sequence<usize>,
    #[define(E)] _dtype: ElemType,
) {
    if !columns.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let spatial = comptime![kernel.len()];
    let (_, position) = decompose_linear(ABSOLUTE_POS, &columns_shape);
    let channels = input.shape(spatial + 1);
    let channel = position[spatial + 1] % channels;
    let mut tap = position[spatial + 1] / channels;
    let mut offset = position[0] * input.stride(0) + channel * input.stride(spatial + 1);
    let mut valid = true;
    #[unroll]
    for i in 0..spatial {
        let axis = spatial - 1 - i;
        let kernel_position = tap % kernel[axis];
        tap /= kernel[axis];
        let source = position[axis + 1] * stride[axis] + kernel_position * dilation[axis];
        valid = valid && source >= padding[axis];
        let source = source - padding[axis];
        valid = valid && source < input.shape(axis + 1);
        offset += source * input.stride(axis + 1);
    }

    let mut value = E::zero();
    if valid {
        value = input[offset];
    }
    columns.write(ABSOLUTE_POS, value);
}

/// Scatter-add [`im2col`]'s adjoint in one kernel. One thread owns one input
/// element and sums every tap that read it, so overlapping taps do not race.
fn col2im<const N: usize>(
    columns: CubeTensor,
    out_shape: &[usize],
    input_shape: &Shape,
    kernel_shape: &[usize],
    options: &ConvOptions<N>,
) -> CubeTensor {
    let channels = input_shape[N + 1]; // NHWC: [batch, ..spatial, C]
    let taps = kernel_shape.iter().product::<usize>();
    let dtype = columns.dtype;
    let accumulation_dtype = match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        _ => dtype,
    };
    let client = columns.client.clone();

    let columns = into_contiguous_aligned(columns);
    let grad_in = empty_device_dtype(
        client.clone(),
        columns.device.clone(),
        input_shape.clone(),
        dtype,
    );

    let num_elems = grad_in.meta.num_elements();
    let cube_dim = CubeDim::new(&client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&client, num_elems, cube_dim);
    let pad = options.padding_begin();

    let mut kernel = SequenceArg::new();
    let mut stride = SequenceArg::new();
    let mut padding = SequenceArg::new();
    let mut dilation = SequenceArg::new();
    let mut out_spatial = SequenceArg::new();
    for axis in 0..N {
        kernel.push(kernel_shape[axis]);
        stride.push(options.stride[axis]);
        padding.push(pad[axis]);
        dilation.push(options.dilation[axis]);
        out_spatial.push(out_shape[axis]);
    }

    unsafe {
        col2im_kernel::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            address_type!(columns, grad_in),
            columns.into_tensor_arg(),
            grad_in.clone().into_linear_view(),
            shape_divmod(&grad_in),
            kernel,
            stride,
            padding,
            dilation,
            out_spatial,
            taps,
            channels,
            [
                dtype_to_storage_type(dtype),
                dtype_to_storage_type(accumulation_dtype),
            ],
        );
    }

    grad_in
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn col2im_kernel<E: Numeric, A: Numeric>(
    columns: &Tensor<E>,
    mut grad_in: LinearViewMut<'_, E>,
    grad_in_shape: Sequence<FastDivmod<usize>>,
    kernel: Sequence<usize>,
    stride: Sequence<usize>,
    pad: Sequence<usize>,
    dilation: Sequence<usize>,
    out_spatial: Sequence<usize>,
    taps: usize,
    channels: usize,
    #[define(E, A)] _dtypes: [ElemType; 2],
) {
    if !grad_in.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let spatial = comptime![kernel.len()];
    let (_, pos) = decompose_linear(ABSOLUTE_POS, &grad_in_shape);
    let n = pos[0];
    let cin = pos[spatial + 1];
    let mut val = A::zero();

    for tap in 0..taps {
        // Row-major over `[batch, ..out_spatial]`, matching `reshape_input`.
        let mut m = n;
        let mut hit = true;

        #[unroll]
        for axis in 0..spatial {
            let mut divisor = 1;
            #[unroll]
            for later in 0..spatial {
                if later > axis {
                    divisor *= kernel[later];
                }
            }
            let offset = (tap / divisor) % kernel[axis];
            let numerator = pos[axis + 1] + pad[axis];
            let shift = offset * dilation[axis];
            let fits = numerator >= shift;
            let numerator = numerator - shift;
            let step = stride[axis];
            let out_pos = numerator / step;
            hit = hit && fits && out_pos * step == numerator && out_pos < out_spatial[axis];
            m = m * out_spatial[axis] + out_pos;
        }

        if hit {
            // Through the strides: a pitched allocator pads each matmul output
            // row, so a row is not always `taps * channels` elements long.
            val += A::cast_from(
                columns[m * columns.stride(0) + (tap * channels + cin) * columns.stride(1)],
            );
        }
    }

    grad_in.write(ABSOLUTE_POS, E::cast_from(val));
}

/// The gradient with respect to a dense convolution's weight, as one matmul
/// over the input's columns.
///
/// `conv_weight_grad_no_groups` computes the same thing by convolving the input
/// *by the output gradient*, which makes the gradient the kernel — so the
/// convolution it submits has a kernel the size of the whole feature map and
/// channels numbering only the batch. That is the worst shape a direct
/// convolution kernel can be handed, and on a device whose dtype has no
/// accelerated matmul it is the only candidate that does not decline. A 3x3
/// over a 128x128 map at batch 4 takes about 20 ms that way, against 1.5 ms for
/// the *data* gradient of the same convolution.
///
/// Laid out as columns it is `[c_out, m] @ [m, ..kernel * c_in]`, a matmul —
/// and one whose contraction is enormous against a small output, so
/// [`split_count`] applies for the same reason it does to the pointwise case.
///
/// The cost is the materialisation: see [`im2col`]. That is the trade this
/// makes, and it is why it is offered to autotune rather than taken as a rule.
pub fn wgrad_im2col<const N: usize>(
    input: CubeTensor,
    out_grad: CubeTensor,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    let rank = input.meta.num_dims();
    let dim_c = rank - 1;

    // Both declines read fields the autotune key holds exactly. A decline that
    // turned on the spatial dimensions or the batch would be unsound: the key
    // anchors those, so a shape that declines can share a key with one that did
    // not, and the tuner unwraps whatever it already picked.
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }
    // A pointwise convolution's columns are a copy of the input feeding the
    // matmul `wgrad_im2col_1x1` already runs without one, so this can only lose.
    if check_pointwise(&weight_shape[1..dim_c], &options).is_ok() {
        return Err(ConvSetupError::Unknown);
    }

    let out_channels = weight_shape[0];
    let in_channels = input.meta.shape()[dim_c];
    let kernel_shape = weight_shape[1..dim_c].to_vec();
    let out_shape = out_grad.meta.shape()[1..dim_c].to_vec();

    let cols = kernel_shape.iter().product::<usize>() * in_channels;
    let rows = out_grad.meta.shape()[..dim_c].iter().product::<usize>();

    let columns = im2col::<N>(input, &out_shape, &kernel_shape, &options)?;
    // `[batch, ..out spatial, cols]` -> `[m, cols]`. Free: only leading
    // dimensions merge, and they are dense in that order.
    let columns = reshape(columns, Shape::new([rows, cols]));

    let out_grad = reshape_input(out_grad); // [m, c_out]
    let dtype = out_grad.dtype;

    // The uncut form, which every way of bowing out of the cut below ends in
    // rather than in an `Err`. What the cut turns on is `rows`, and the key
    // holds the spatial dimensions and the batch only anchored: a shape that
    // declines can share a key with one that did not, and the tuner unwraps
    // whatever it already picked, so declining on a cached hit aborts.
    let uncut = |columns: CubeTensor, out_grad: CubeTensor| {
        let out_grad = swap_dims(out_grad, 0, 1); // [c_out, m]
        matmul(out_grad, columns, None, MatmulStrategy::default(), dtype)
    };

    let grad = match split_count(rows) {
        // `[split, c_out, m / split] @ [split, m / split, cols]`, the partials
        // summed. Every reshape is free, as in `wgrad_im2col_1x1_split`.
        Some(split) => {
            let per = rows / split;
            let cut = reshape(columns.clone(), Shape::new([split, per, cols]));
            let grad = reshape(out_grad.clone(), Shape::new([split, per, out_channels]));
            let grad = swap_dims(grad, 1, 2);

            let partials = matmul(grad, cut, None, MatmulStrategy::default(), dtype)
                .ok()
                .and_then(|partials| {
                    reduce_dim(
                        partials,
                        None,
                        0,
                        KernelReduceStrategy::default(),
                        ReduceOperationConfig::Sum,
                    )
                    .ok()
                });

            match partials {
                Some(grad) => grad,
                None => uncut(columns, out_grad)?,
            }
        }
        None => uncut(columns, out_grad)?,
    };

    // `[c_out, ..kernel * c_in]` -> `[c_out, ..kernel, c_in]`, which is the
    // order the columns were laid out in.
    Ok(reshape(grad, weight_shape))
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    /// Checks columns `[batch, ...output_spatial, kernel_taps * channels]`
    /// against host indexing of `[batch, ...input_spatial, channels]`.
    fn assert_columns<const N: usize>(shape: &[usize], kernel: &[usize], options: ConvOptions<N>) {
        let device = CubeDevice::cuda(0).unwrap();
        let values = (0..shape.iter().product::<usize>())
            .map(|i| i as f32 + 1.0)
            .collect();
        let mut input = crate::ops::from_data(TensorData::new(values, shape), &device);
        if const { N > 1 } {
            input = swap_dims(input, 1, 2);
        }
        let shape = input.meta.shape().to_vec();
        let values = crate::ops::into_data_sync(input.clone())
            .try_to_vec::<f32>()
            .unwrap();
        let out_shape = calculate_conv_output_sizes(
            kernel,
            &options.stride,
            &options.padding,
            &options.dilation,
            &shape[1..N + 1],
        );
        let columns = im2col(input, &out_shape, kernel, &options).unwrap();
        let column_shape = columns.meta.shape().to_vec();
        let actual = crate::ops::into_data_sync(columns)
            .try_to_vec::<f32>()
            .unwrap();
        let channels = shape[N + 1];

        for (index, actual) in actual.into_iter().enumerate() {
            let mut rest = index;
            let mut position = vec![0; N + 2];
            for axis in (0..N + 2).rev() {
                position[axis] = rest % column_shape[axis];
                rest /= column_shape[axis];
            }
            let channel = position[N + 1] % channels;
            let mut tap = position[N + 1] / channels;
            let mut source = vec![0; N];
            let mut valid = true;
            for axis in (0..N).rev() {
                let offset = tap % kernel[axis];
                tap /= kernel[axis];
                let coordinate = (position[axis + 1] * options.stride[axis]
                    + offset * options.dilation[axis]) as isize
                    - options.padding[axis].0 as isize;
                valid &= coordinate >= 0 && coordinate < shape[axis + 1] as isize;
                source[axis] = coordinate.max(0) as usize;
            }
            let mut source_index = position[0];
            for axis in 0..N {
                source_index = source_index * shape[axis + 1] + source[axis];
            }
            source_index = source_index * channels + channel;
            let expected = if valid { values[source_index] } else { 0.0 };
            assert_eq!(actual, expected, "column element {index}");
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn columns_handle_coordinates_larger_than_u32() {
        assert_columns(
            &[1, 1, 1],
            &[3],
            ConvOptions::new_with_padding([1], [(0, 1usize << 32)], [1usize << 31], 1),
        );
        assert_columns(
            &[1, 1, 1],
            &[2],
            ConvOptions::new_with_padding([1], [(0, 1usize << 32)], [1usize << 32], 1),
        );
        assert_columns(
            &[1, 1, 1],
            &[1],
            ConvOptions::new_with_padding([1usize << 31], [(0, 1usize << 32)], [1], 1),
        );
        assert_columns(
            &[1, 1, 1],
            &[2],
            ConvOptions::new_with_padding([1], [(1usize << 32, 0)], [1usize << 32], 1),
        );
    }

    #[test]
    fn columns_reject_coordinates_larger_than_usize() {
        let device = CubeDevice::cuda(0).unwrap();
        let input = crate::ops::from_data(TensorData::new(vec![1.0_f32], [1, 1, 1]), &device);
        let options = ConvOptions::new([1], [0], [usize::MAX], 1);

        assert!(matches!(
            im2col(input, &[1], &[3], &options),
            Err(ConvSetupError::InvalidConfig(_))
        ));
    }

    #[test]
    fn single_pass_columns_match_host_indexing() {
        assert_columns(
            &[2, 9, 3],
            &[4],
            ConvOptions::new_with_padding([2], [(1, 2)], [2], 1),
        );
        assert_columns(
            &[2, 7, 9, 3],
            &[2, 3],
            ConvOptions::new_with_padding([2, 3], [(1, 2), (2, 0)], [2, 1], 1),
        );
        assert_columns(
            &[1, 6, 7, 5, 2],
            &[2, 3, 2],
            ConvOptions::new([2, 1, 2], [1, 0, 1], [1, 2, 1], 1),
        );
    }
}
