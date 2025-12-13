use burn_backend::{TensorMetadata, ops::ConvOptions};
use burn_std::{Shape, Slice};
use cubek::convolution::components::ConvSetupError;

use crate::{
    CubeRuntime,
    kernel::{conv::base::conv_forward_nhwc, slice, slice_assign},
    ops::{numeric::empty_device_optimized_dtype, swap_dims},
    tensor::CubeTensor,
};

/// Calculate the [2D convolution](crate::ops::ModuleOps::conv2d) backward pass, returning the gradient for `weight`.
pub fn conv_weight_backward_fallback<R: CubeRuntime, const N_DIM: usize>(
    input: CubeTensor<R>,
    output_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N_DIM>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    match options.groups == 1 {
        true => conv_weight_grad_no_groups::<R, N_DIM>(input, output_grad, weight_shape, options),
        false => conv_weight_grad_groups::<R, N_DIM>(input, output_grad, weight_shape, options),
    }
}

fn conv_weight_grad_no_groups<R: CubeRuntime, const N_DIM: usize>(
    input: CubeTensor<R>,
    output_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N_DIM>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let dim_c = input.rank() - 1;

    let input_swapped = swap_dims(input, 0, dim_c);
    let out_grad_swapped = swap_dims(output_grad, 0, dim_c);
    let weight_grad_swapped = conv_forward_nhwc(
        input_swapped,
        out_grad_swapped,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
        Default::default(),
    )?;
    let mut weight_grad = swap_dims(weight_grad_swapped, 0, dim_c);
    if weight_grad.shape != weight_shape {
        let ranges = weight_shape.iter().map(|&s| 0..s).collect::<Vec<_>>();
        weight_grad = slice(weight_grad, &ranges);
    }

    Ok(weight_grad)
}

#[allow(clippy::single_range_in_vec_init, reason = "False positive")]
fn conv_weight_grad_groups<R: CubeRuntime, const N_DIM: usize>(
    input: CubeTensor<R>,
    output_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N_DIM>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let mut weight_grad = empty_device_optimized_dtype(
        input.client.clone(),
        input.device.clone(),
        weight_shape.clone(),
        input.dtype,
    );

    let dim_c = input.rank() - 1;

    let channels_out = weight_shape.dims[0];
    let increment_co = channels_out / options.groups;

    let input_swapped = swap_dims(input, 0, dim_c);
    let output_grad_swapped = swap_dims(output_grad, 0, dim_c);

    let kernel_size = &weight_shape[1..dim_c];
    let kernel_size_slice = kernel_size.iter().map(|&s| 0..s).collect::<Vec<_>>();
    let increment_ci = weight_grad.shape.dims[dim_c];

    for g in 0..options.groups {
        let start_idx_ci = g * increment_ci;
        let end_idx_ci = (g + 1) * increment_ci;
        let start_idx_co = g * increment_co;
        let end_idx_co = (g + 1) * increment_co;

        let input = slice(input_swapped.clone(), &[start_idx_ci..end_idx_ci]);
        let grad = slice(output_grad_swapped.clone(), &[start_idx_co..end_idx_co]);

        let weight_grad_tmp = conv_forward_nhwc(
            input,
            grad,
            None,
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
            Default::default(),
        )?;
        let mut weight_grad_tmp = swap_dims(weight_grad_tmp, 0, dim_c);
        let kernel_size_tmp = &weight_grad_tmp.shape.dims[1..dim_c];

        if kernel_size != kernel_size_tmp {
            let mut slices = vec![0..increment_co];
            slices.extend(kernel_size_slice.clone());
            slices.push(0..increment_ci);
            weight_grad_tmp = slice(weight_grad_tmp, &slices);
        }

        let mut slices = vec![start_idx_co..end_idx_co];
        slices.extend(kernel_size_slice.clone());
        slices.push(0..increment_ci);
        let slices = slices.into_iter().map(Slice::from).collect::<Vec<_>>();

        weight_grad = slice_assign(weight_grad, &slices, weight_grad_tmp);
    }

    Ok(weight_grad)
}
