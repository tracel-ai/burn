use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, ops::ConvOptions};
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
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let rank = input.meta.num_dims();
    let dim_c = rank - 1;

    let out_channels = weight.meta.shape()[0];

    check_pointwise(&weight.meta.shape()[1..dim_c], &options)?;

    // A pointwise convolution's output has the input's spatial shape.
    let mut split_m = vec![input.meta.shape()[0]];
    split_m.extend(input.meta.shape()[1..dim_c].iter().copied());

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
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let pointwise = kernel_shape.iter().all(|size| *size == 1)
        && options.stride.iter().all(|stride| *stride == 1)
        && options.padding.iter().all(|padding| *padding == 0)
        && options.dilation.iter().all(|dilation| *dilation == 1);

    match pointwise {
        true => Ok(()),
        false => Err(ConvSetupError::Unknown),
    }
}

/// Drops a pointwise weight's unit kernel dimensions, giving `[C_out, C_in]`.
///
/// Rewriting the metadata rather than reshaping keeps a padded channel stride,
/// so a weight the pitched allocator already aligned for TMA is not copied to
/// say so. One that is not gets a pitched copy here rather than a second kernel
/// inside the matmul.
fn reshape_weight<R: CubeRuntime>(mut weight: CubeTensor<R>) -> CubeTensor<R> {
    let dim_c = weight.meta.num_dims() - 1;
    let strides = [weight.meta.strides()[0], weight.meta.strides()[dim_c]];
    let shape = [weight.meta.shape()[0], weight.meta.shape()[dim_c]];

    *weight.meta = Metadata::new(shape, strides);

    match strides[1] {
        1 => weight,
        _ => into_contiguous_aligned(weight),
    }
}

/// The gradient of a pointwise convolution with respect to its input, as one
/// matmul.
///
/// `grad_in[(n, h, w), c_in] = sum over c_out of grad_out[(n, h, w), c_out] *
/// weight[c_out, c_in]`. The fallback computes the same thing as a transposed
/// convolution, which has no NHWC path and falls back to a naive kernel on a
/// device with no accelerated matmul for the dtype.
pub fn dgrad_im2col_1x1<R: CubeRuntime, const N: usize>(
    out_grad: CubeTensor<R>,
    weight: CubeTensor<R>,
    input_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
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
pub fn wgrad_im2col_1x1<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    out_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
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
