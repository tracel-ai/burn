use burn_backend::{
    DType,
    ops::{ConvOptions, conv::calculate_conv_output_sizes},
};
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
    let max_simultaneous = (max_cube_count / cube_count_per_batch).min(batch_size);
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

    let rank = input.shape.num_dims();
    let dim_c = rank - 1;

    let batch_size = input.shape[0];
    let in_channels = input.shape[dim_c];
    let in_shape = &input.shape[1..dim_c];
    let out_channels = weight.shape[0];
    let kernel_shape = &weight.shape[1..dim_c];

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
    let weight = if weight.strides[dim_c] != 1 {
        // Remove kernel dims so padded dim is channels
        weight.shape.dims = vec![out_channels, in_channels]; // [N, K]
        weight.strides = vec![weight.strides[0], weight.strides[dim_c]];
        // Pitched contiguous to skip running another kernel for TMA
        into_contiguous_aligned(weight)
    } else {
        // Already compatible, skip initial reshape
        weight.shape.dims = vec![out_channels, in_channels]; // [N, K]
        weight.strides = vec![weight.strides[0], 1];
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
fn reshape_input<R: CubeRuntime>(mut input: CubeTensor<R>) -> CubeTensor<R> {
    let rank = input.shape.num_dims();
    let dim_c = rank - 1;
    let dtype = input.dtype;

    let batch_size = input.shape[0];
    let in_c: usize = input.shape[dim_c];
    let in_shape = input.shape[1..dim_c].to_vec();

    if !is_spatial_contiguous(&input.shape, &input.strides) {
        let contiguous =
            into_contiguous_pitched(&input.client, &input.as_handle_ref(), dtype.into())
                .expect("Kernel to never fail");
        input = from_handle(&input.client, &input.device, contiguous, dtype);
    }
    input.shape.dims = vec![batch_size * in_shape.iter().product::<usize>(), in_c]; // [M, K]
    input.strides = vec![input.strides[dim_c - 1], input.strides[dim_c]];
    input
}

fn is_spatial_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    let rank = shape.len();

    let mut ordered = strides.to_vec();
    ordered.sort();
    if ordered != strides {
        return false;
    }

    for i in (1..rank - 2).rev() {
        if strides[i + 1] * shape[i + 1] != strides[i] {
            return false;
        }
    }
    true
}

fn from_handle<R: CubeRuntime>(
    client: &ComputeClient<R>,
    device: &R::Device,
    handle: TensorHandle<R>,
    dtype: DType,
) -> CubeTensor<R> {
    CubeTensor::new(
        client.clone(),
        handle.handle,
        handle.shape.into(),
        device.clone(),
        handle.strides,
        dtype,
    )
}
