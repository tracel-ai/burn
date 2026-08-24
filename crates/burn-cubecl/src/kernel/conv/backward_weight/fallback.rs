use burn_backend::{TensorMetadata, ops::ConvOptions};
use burn_std::{Shape, Slice};
use cubek::convolution::components::ConvSetupError;

use crate::{
    CubeRuntime,
    kernel::{conv::base::conv_forward_nhwc, into_contiguous_aligned, slice, slice_assign},
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

    // The depthwise case is separated out because the general one costs a
    // kernel *per group*, and a depthwise convolution's group count is its
    // channel count — hundreds or thousands of launches for one gradient.
    match options.groups {
        1 => conv_weight_grad_no_groups::<R, N_DIM>(input, output_grad, weight_shape, options),
        groups if groups == in_channels => {
            conv_weight_grad_depthwise::<R, N_DIM>(input, output_grad, weight_shape, options)
        }
        _ => conv_weight_grad_groups::<R, N_DIM>(input, output_grad, weight_shape, options),
    }
}

/// The weight gradient of a *depthwise* convolution, as one grouped convolution.
///
/// [`conv_weight_grad_groups`] computes a grouped weight gradient by convolving
/// each group on its own, which is correct and costs one kernel launch per
/// group. When the convolution is depthwise there is one group per channel, so
/// a single block of EfficientNet's later stages submits thousands of launches
/// to differentiate one 3x3 — each over a single channel, which is far too
/// little work to cover the launch that carries it. The arithmetic is trivial;
/// only the shape of the loop is not.
///
/// It does not have to be a loop. [`conv_weight_grad_no_groups`] already frames
/// a weight gradient as a convolution of the *input* by the *output gradient*,
/// with batch and channel swapped so that each input channel is an image and
/// each output channel a filter. The same framing survives being made
/// depthwise, provided the channel that must not mix is the one the groups are
/// cut along:
///
/// - the input becomes a single image of `channels * batch` channels, ordered
///   so that a channel's batch elements are adjacent — the one materialised
///   copy this path pays, and it is one pass over the activation;
/// - the output gradient becomes `channels * multiplier` filters of `batch`
///   channels each, which is a metadata swap;
/// - `groups = channels` then cuts the image so that filter `o` sees the batch
///   of input channel `o / multiplier` and nothing else, which is exactly the
///   sum a depthwise weight gradient is.
///
/// Restricted to `groups == in_channels`. A general grouped convolution carries
/// more than one input channel per group, and those channels have to stay
/// *apart* in the result rather than being summed over — which is the one thing
/// this framing cannot express, since a filter sums over every channel of its
/// group. Those keep the loop.
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

    // `[N, ..spatial, C]` -> `[..spatial, C, N]`, so that a channel's batch
    // elements end up adjacent and the merge below is a reshape rather than a
    // second copy. This is the only data movement the path adds, and it reads
    // and writes the activation once.
    let mut rolled_axes: Vec<usize> = (1..rank).collect();
    rolled_axes.push(0);
    let rolled = into_contiguous_aligned(permute(input, &rolled_axes));

    // -> `[1, ..spatial, C * N]`. One image, whose channels are every (channel,
    // batch) pair. Free: only the last two dimensions merge, and they are
    // adjacent and contiguous after the roll.
    let mut image_shape = vec![1];
    image_shape.extend(rolled.meta.shape()[..rank - 2].iter().copied());
    image_shape.push(channels * batch_size);
    let image = reshape(rolled, image_shape.into());

    // `[N, ..out spatial, C_out]` -> `[C_out, ..out spatial, N]`: the gradient
    // read as `C_out` filters of `N` channels. A metadata swap, as it is in the
    // no-groups path.
    let filter = swap_dims(output_grad, 0, dim_c);

    // Stride and dilation trade places, because the gradient is being used as
    // the kernel: the step between the *kernel's* taps is the convolution's
    // stride, and the step between the *image's* is its dilation.
    let weight_grad = conv_forward_nhwc(
        image,
        filter,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, channels),
        Default::default(),
    )?;

    // `[1, ..kernel, C_out]` -> `[C_out, ..kernel, 1]`, which is the weight's
    // own NHWC shape: the batch axis is a unit axis here, so this swap is the
    // same one the no-groups path ends with.
    let mut weight_grad = swap_dims(weight_grad, 0, dim_c);

    // The convolution's output can overhang the kernel when a stride does not
    // divide the input evenly, exactly as in the no-groups path.
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
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
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
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
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
