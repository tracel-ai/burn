use burn_backend::{TensorMetadata, ops::ConvOptions};
use burn_std::{Shape, Slice};
use cubek::convolution::components::ConvSetupError;

use crate::{
    CubeRuntime,
    kernel::{conv::base::conv_forward_nhwc, slice, slice_assign},
    ops::{numeric::empty_device_dtype, permute, reshape, swap_dims},
    tensor::CubeTensor,
};

/// Calculate the convolution backward pass with regard to the weight gradients.
pub fn conv_weight_backward_fallback<R: CubeRuntime, const N_DIM: usize>(
    input: CubeTensor<R>,
    output_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N_DIM>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let in_channels = input.meta.shape()[input.rank() - 1];

    // Depthwise is separated out because the general grouped path costs a
    // kernel per group, and a depthwise convolution has one group per channel.
    match options.groups {
        1 => conv_weight_grad_no_groups::<R, N_DIM>(input, output_grad, weight_shape, options),
        groups if groups == in_channels => {
            conv_weight_grad_depthwise::<R, N_DIM>(input, output_grad, weight_shape, options)
        }
        _ => conv_weight_grad_groups::<R, N_DIM>(input, output_grad, weight_shape, options),
    }
}

/// The weight gradient of a depthwise convolution, as one grouped convolution.
///
/// [`conv_weight_grad_groups`] launches a kernel per group, and a depthwise
/// convolution has one group per channel — a single block of EfficientNet's
/// later stages submits thousands of launches, each over one channel, to
/// differentiate one 3x3.
///
/// [`conv_weight_grad_no_groups`] already frames a weight gradient as a
/// convolution of the input by the output gradient, each input channel an image
/// and each output channel a filter. That framing survives being made depthwise
/// by folding the batch into the channels: the input becomes one image of
/// `channels * batch` channels ordered so a channel's batch elements are
/// adjacent, the gradient becomes `channels * multiplier` filters of `batch`
/// channels each, and `groups = channels` cuts the image so filter `o` sees the
/// batch of input channel `o / multiplier` and nothing else.
///
/// Restricted to `groups == in_channels`. A group carrying several input
/// channels needs them kept apart in the result, and a filter sums over every
/// channel of its group.
fn conv_weight_grad_depthwise<R: CubeRuntime, const N_DIM: usize>(
    input: CubeTensor<R>,
    output_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N_DIM>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let rank = input.rank();
    let dim_c = rank - 1;

    let batch_size = input.meta.shape()[0];
    let channels = input.meta.shape()[dim_c];

    // `[N, ..spatial, C]` -> `[1, ..spatial, C * N]`: one image whose channels
    // are every (channel, batch) pair. The permutation is not expressible in
    // strides, so this is the single copy the path adds — one pass over the
    // activation.
    let mut rolled_axes = (1..rank).collect::<Vec<_>>();
    rolled_axes.push(0);
    let mut image_shape = vec![1];
    image_shape.extend(input.meta.shape()[1..dim_c].iter().copied());
    image_shape.push(channels * batch_size);
    let image = reshape(permute(input, &rolled_axes), image_shape.into());

    // `[N, ..out spatial, C_out]` -> `[C_out, ..out spatial, N]`: the gradient
    // read as `C_out` filters of `N` channels. A metadata swap, as in the
    // no-groups path.
    let filter = swap_dims(output_grad, 0, dim_c);

    // Stride and dilation trade places, because the gradient is the kernel: the
    // step between the kernel's taps is the convolution's stride, and the step
    // between the image's is its dilation.
    let weight_grad = conv_forward_nhwc(
        image,
        filter,
        None,
        ConvOptions::new_with_padding(options.dilation, options.padding, options.stride, channels),
        Default::default(),
    )?;

    // `[1, ..kernel, C_out]` -> `[C_out, ..kernel, 1]`, the weight's own NHWC
    // shape: the batch axis is a unit axis here.
    let mut weight_grad = swap_dims(weight_grad, 0, dim_c);

    // The convolution's output can overhang the kernel when a stride does not
    // divide the input evenly, as in the no-groups path.
    if weight_grad.shape() != weight_shape {
        let ranges = weight_shape.iter().map(|&s| 0..s).collect::<Vec<_>>();
        weight_grad = slice(weight_grad, &ranges);
    }

    Ok(weight_grad)
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
        ConvOptions::new_with_padding(options.dilation, options.padding, options.stride, 1),
        Default::default(),
    )?;
    let mut weight_grad = swap_dims(weight_grad_swapped, 0, dim_c);
    if weight_grad.shape() != weight_shape {
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
    let mut weight_grad = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        weight_shape.clone(),
        input.dtype,
    );

    let dim_c = input.rank() - 1;

    let channels_out = weight_shape[0];
    let increment_co = channels_out / options.groups;

    let input_swapped = swap_dims(input, 0, dim_c);
    let output_grad_swapped = swap_dims(output_grad, 0, dim_c);

    let kernel_size = &weight_shape[1..dim_c];
    let kernel_size_slice = kernel_size.iter().map(|&s| 0..s).collect::<Vec<_>>();
    let increment_ci = weight_grad.meta.shape()[dim_c];

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
            ConvOptions::new_with_padding(options.dilation, options.padding, options.stride, 1),
            Default::default(),
        )?;
        let mut weight_grad_tmp = swap_dims(weight_grad_tmp, 0, dim_c);
        let kernel_size_tmp = &weight_grad_tmp.meta.shape()[1..dim_c];

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
