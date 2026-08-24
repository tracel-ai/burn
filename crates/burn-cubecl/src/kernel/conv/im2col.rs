use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{
    DType,
    ops::{ConvOptions, conv::calculate_conv_output_sizes},
};
use burn_std::{Metadata, Shape};
use core::iter;
use cubecl::{
    prelude::*,
    std::tensor::{TensorHandle, into_contiguous_pitched},
};
use cubek::convolution::components::ConvSetupError;

use crate::{
    CubeRuntime,
    kernel::{
        AddOp, into_contiguous_aligned, launch_binop,
        matmul::{MatmulStrategy, matmul},
        utils::split_dim,
    },
    ops::{reshape, swap_dims},
    tensor::CubeTensor,
};

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

pub fn conv_im2col_1x1<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    mut weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let rank = input.meta.num_dims();
    let dim_c = rank - 1;

    let batch_size = input.meta.shape()[0];
    let in_channels = input.meta.shape()[dim_c];
    let in_shape = &input.meta.shape()[1..dim_c];
    let out_channels = weight.meta.shape()[0];
    let kernel_shape = &weight.meta.shape()[1..dim_c];

    if kernel_shape.iter().any(|s| *s != 1) {
        return Err(ConvSetupError::Unknown);
    }

    let out_shape = calculate_conv_output_sizes(
        kernel_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        in_shape,
    );

    let mut split_m = vec![batch_size];
    split_m.extend(out_shape.iter().copied());

    if kernel_shape.iter().any(|it| *it != 1) || in_shape != out_shape {
        return Err(ConvSetupError::Unknown);
    }

    let input = reshape_input(input); // [(NHW), C] : [M, K]
    let dtype = input.dtype;

    // Efficient permutation that takes the stride required for TMA into account
    let weight = if weight.meta.strides()[dim_c] != 1 {
        // Remove kernel dims so padded dim is channels
        *weight.meta = Metadata::new(
            [out_channels, in_channels], // [N, K]
            [weight.meta.strides()[0], weight.meta.strides()[dim_c]],
        );
        // Pitched contiguous to skip running another kernel for TMA
        into_contiguous_aligned(weight)
    } else {
        // Already compatible, skip initial reshape
        *weight.meta = Metadata::new([out_channels, in_channels], [weight.meta.strides()[0], 1]);
        weight
    };

    // Permute to N-major, while keeping memory layout K-major. K-major for both sides is the most
    // efficient for matmul, and allows skipping a contiguous kernel
    let weight = swap_dims(weight, 0, 1); // [K, N]

    let out = matmul(input, weight, None, MatmulStrategy::default(), dtype)?; // [M, N]

    // Skip reshape to avoid potential `into_contiguous`. We're only splitting dims so it's safe.
    let mut out = split_dim(out, 0, &split_m); // [N, H, W, C]

    if let Some(bias) = bias {
        let mut bias_shape = iter::repeat_n(1, rank - 1).collect::<Vec<_>>();
        bias_shape.push(out_channels);
        let bias = reshape(bias, bias_shape.into());
        out = launch_binop::<R, AddOp>(out, bias);
    }

    Ok(out)
}

/// Reshapes NHWC input to [(N, H, W), C]
fn reshape_input<R: CubeRuntime>(input: CubeTensor<R>) -> CubeTensor<R> {
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

fn from_handle<R: CubeRuntime>(
    client: ComputeClient<R>,
    device: R::Device,
    handle: TensorHandle<R>,
    dtype: DType,
) -> CubeTensor<R> {
    CubeTensor::new(
        client.clone(),
        handle.handle,
        *handle.metadata,
        device.clone(),
        dtype,
    )
}

/// Whether a convolution is the 1x1 case both gradient paths below reduce to a
/// single matmul.
///
/// A 1x1 convolution that neither strides nor pads is a per-pixel linear map:
/// every output pixel reads exactly the input pixel under it. `im2col` for that
/// case is the identity, which is why the forward's `conv_im2col_1x1` gets away
/// with a reshape and a matmul, and why both of its gradients do too.
///
/// The shapes are NHWC, as everything in `conv/base.rs` is by the time it
/// dispatches.
fn is_pointwise<const N: usize>(
    kernel_shape: &[usize],
    in_shape: &[usize],
    out_shape: &[usize],
    options: &ConvOptions<N>,
) -> Result<(), ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    // Stride and padding are implied by the shapes agreeing — a strided or
    // padded 1x1 does not produce an output the size of its input — but they
    // are checked directly so the reason a shape mismatched is not guessed at.
    if kernel_shape.iter().any(|size| *size != 1)
        || options.stride.iter().any(|stride| *stride != 1)
        || options.padding.iter().any(|padding| *padding != 0)
        || options.dilation.iter().any(|dilation| *dilation != 1)
        || in_shape != out_shape
    {
        return Err(ConvSetupError::Unknown);
    }

    Ok(())
}

/// The gradient with respect to a 1x1 convolution's *input*, as one matmul.
///
/// `grad_in[(n, h, w), c_in] = grad_out[(n, h, w), c_out] * weight[c_out, c_in]`
/// — the same `[M, K] @ [K, N]` the forward does, with the weight read in the
/// other direction and `c_out` as the reduced axis.
///
/// The fallback this stands beside computes the same thing as a *transposed
/// convolution*, and on a device with no accelerated matmul for the dtype that
/// is a naive kernel; it also has no NHWC path, so it permutes both ways around
/// itself on top. Neither cost is inherent to the arithmetic.
pub fn dgrad_im2col_1x1<R: CubeRuntime, const N: usize>(
    out_grad: CubeTensor<R>,
    mut weight: CubeTensor<R>,
    input_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let rank = out_grad.meta.num_dims();
    let dim_c = rank - 1;

    let batch_size = out_grad.meta.shape()[0];
    let out_channels = weight.meta.shape()[0];
    let in_channels = weight.meta.shape()[dim_c];
    let kernel_shape = &weight.meta.shape()[1..dim_c];
    let out_shape = &out_grad.meta.shape()[1..dim_c];
    let in_shape = &input_shape[1..dim_c];

    is_pointwise(kernel_shape, in_shape, out_shape, &options)?;

    let mut split_m = vec![batch_size];
    split_m.extend(out_shape.iter().copied());

    let out_grad = reshape_input(out_grad); // [(NHW), C_out] : [M, K]
    let dtype = out_grad.dtype;

    // The weight arrives as `[C_out, 1, .., 1, C_in]` and is wanted as
    // `[K, N] = [C_out, C_in]`, which is the same buffer with the unit kernel
    // dimensions dropped. Dropping them by rewriting the metadata rather than
    // reshaping keeps a padded `C_in` stride intact, so a weight that is
    // already aligned is not copied to say so.
    let weight = if weight.meta.strides()[dim_c] != 1 {
        *weight.meta = Metadata::new(
            [out_channels, in_channels],
            [weight.meta.strides()[0], weight.meta.strides()[dim_c]],
        );
        into_contiguous_aligned(weight)
    } else {
        *weight.meta = Metadata::new([out_channels, in_channels], [weight.meta.strides()[0], 1]);
        weight
    };

    // No transpose here, unlike the forward: it wants `[K, N]` and the weight's
    // own order *is* `[C_out, C_in]`. The forward has to swap because it
    // reduces over `C_in` and this reduces over `C_out`.
    let out = matmul(out_grad, weight, None, MatmulStrategy::default(), dtype)?; // [M, N]

    // Only splitting a dimension, so this cannot force a copy.
    Ok(split_dim(out, 0, &split_m)) // [N, H, W, C_in]
}

/// The gradient with respect to a 1x1 convolution's *weight*, as one matmul.
///
/// `grad_w[c_out, c_in] = sum over (n, h, w) of grad_out[(n, h, w), c_out] *
/// input[(n, h, w), c_in]` — a `[C_out, M] @ [M, C_in]`, where the reduced axis
/// is every pixel in the batch rather than a channel. That makes it a tall
/// reduction into a small output, which is the shape a tuned matmul is good at
/// and the shape the fallback's "convolve the input by the gradient" framing
/// hides.
pub fn wgrad_im2col_1x1<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    out_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let rank = input.meta.num_dims();
    let dim_c = rank - 1;

    let kernel_shape = &weight_shape[1..dim_c];
    let in_shape = &input.meta.shape()[1..dim_c];
    let out_shape = &out_grad.meta.shape()[1..dim_c];

    is_pointwise(kernel_shape, in_shape, out_shape, &options)?;

    let input = reshape_input(input); // [(NHW), C_in] : [M, N]
    let out_grad = reshape_input(out_grad); // [(NHW), C_out] : [M, K]
    let dtype = out_grad.dtype;

    // `[M, C_out]` read as `[C_out, M]`. A metadata swap, so the matmul reads
    // the gradient K-major rather than a transposed copy of it being made.
    let out_grad = swap_dims(out_grad, 0, 1); // [C_out, M]

    let grad = matmul(out_grad, input, None, MatmulStrategy::default(), dtype)?; // [C_out, C_in]

    // Back to `[C_out, 1, .., 1, C_in]`, which is what the caller permutes to
    // NCHW. Only unit dimensions are being reinserted, so nothing moves.
    Ok(reshape(grad, weight_shape))
}
