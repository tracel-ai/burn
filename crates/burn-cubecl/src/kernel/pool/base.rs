use crate::{
    kernel::{
        into_contiguous_aligned,
        reduce::{KernelReduceStrategy, reduce_dim},
    },
    ops::{base::reshape, numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, Shape, ops::conv::calculate_pool_output_size};
use cubek::pool::{
    definition::{AdaptiveAvgPoolOptions, AvgPoolOptions, MaxPoolOptions, PoolError, PoolMode},
    pool2d, pool2d_backward, pool2d_with_indices, pool2d_with_indices_backward, pool3d,
    pool3d_backward,
};
use cubek::reduce::components::instructions::ReduceOperationConfig;

pub(crate) fn max_pool2d(
    x: CubeTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> CubeTensor {
    let [batch_size, channels, height, width] = x.meta.shape().dims();

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        height,
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        width,
        ceil_mode,
    );

    let x = into_contiguous_aligned(permute_nchw_to_nhwc(x));

    let shape_out = Shape::new([batch_size, size_0, size_1, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, x.dtype);

    let mode = PoolMode::from(MaxPoolOptions::new(
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    ));

    pool2d(
        &output.client,
        x.clone().binding(),
        output.clone().binding(),
        mode,
        dtype_to_storage_type(output.dtype),
    )
    .unwrap_or_else(|e| pool_panic("max_pool2d", &x, e));

    permute_nhwc_to_nchw(output)
}

pub(crate) fn max_pool2d_with_indices(
    x: CubeTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    dtype_indices: DType,
) -> (CubeTensor, CubeTensor) {
    let [batch_size, channels, size_0, size_1] = x.meta.shape().dims();

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        size_0,
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        size_1,
        ceil_mode,
    );

    let x = into_contiguous_aligned(permute_nchw_to_nhwc(x));

    let shape_out = Shape::new([batch_size, size_0, size_1, channels]);
    let output = empty_device_dtype(
        x.client.clone(),
        x.device.clone(),
        shape_out.clone(),
        x.dtype,
    );
    let indices = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, dtype_indices);

    let mode = PoolMode::from(MaxPoolOptions::new(
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    ));

    pool2d_with_indices(
        &output.client,
        x.clone().binding(),
        output.clone().binding(),
        indices.clone().binding(),
        mode,
        dtype_to_storage_type(output.dtype),
    )
    .unwrap_or_else(|e| pool_panic("max_pool2d_with_indices", &x, e));

    let output = permute_nhwc_to_nchw(output);
    let indices = permute_nhwc_to_nchw(indices);
    (output, indices)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn max_pool2d_with_indices_backward(
    x: CubeTensor,
    grad: CubeTensor,
    indices: CubeTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> CubeTensor {
    let [batches, channels, height, width] = x.meta.shape().dims();
    let input = into_contiguous_aligned(permute_nchw_to_nhwc(x));
    let grad = into_contiguous_aligned(permute_nchw_to_nhwc(grad));
    let indices = into_contiguous_aligned(permute_nchw_to_nhwc(indices));

    let out_shape = Shape::new([batches, height, width, channels]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        out_shape,
        input.dtype,
    );

    let mode = PoolMode::from(MaxPoolOptions::new(
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    ));

    pool2d_with_indices_backward(
        &output.client,
        input.clone().binding(),
        grad.clone().binding(),
        indices.clone().binding(),
        output.clone().binding(),
        mode,
        dtype_to_storage_type(output.dtype),
        dtype_to_storage_type(indices.dtype),
    )
    .unwrap_or_else(|e| pool_panic("max_pool2d_with_indices_backward", &input, e));

    permute_nhwc_to_nchw(output)
}

pub(crate) fn avg_pool2d(
    x: CubeTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> CubeTensor {
    let [batch_size, channels, in_h, in_w] = x.meta.shape().dims();
    let dilation = 1;

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation,
        in_h,
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation,
        in_w,
        ceil_mode,
    );

    let x = into_contiguous_aligned(permute_nchw_to_nhwc(x));

    let shape_out = Shape::new([batch_size, size_0, size_1, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, x.dtype);

    let mode = PoolMode::from(AvgPoolOptions::new(
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
    ));

    pool2d(
        &output.client,
        x.clone().binding(),
        output.clone().binding(),
        mode,
        dtype_to_storage_type(output.dtype),
    )
    .unwrap_or_else(|e| pool_panic("avg_pool2d", &x, e));

    permute_nhwc_to_nchw(output)
}

pub(crate) fn avg_pool2d_backward(
    x: CubeTensor,
    grad: CubeTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> CubeTensor {
    let [batches, channels, height, width] = x.meta.shape().dims();
    let input = into_contiguous_aligned(permute_nchw_to_nhwc(x));
    let grad = into_contiguous_aligned(permute_nchw_to_nhwc(grad));

    let out_shape = Shape::new([batches, height, width, channels]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        out_shape,
        input.dtype,
    );

    let mode = PoolMode::from(AvgPoolOptions::new(
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
    ));

    pool2d_backward(
        &output.client,
        input.clone().binding(),
        grad.clone().binding(),
        output.clone().binding(),
        mode,
        dtype_to_storage_type(output.dtype),
    )
    .unwrap_or_else(|e| pool_panic("avg_pool2d_backward", &input, e));

    permute_nhwc_to_nchw(output)
}

/// Average every pixel of every channel into one number: `[b, c, h, w]` to
/// `[b, c, 1, 1]`.
///
/// This is what an adaptive average pool with a `1x1` output *is*, and it is a
/// reduction rather than a pooling problem.
///
/// So the reduction axis becomes the parallel one: the two spatial axes are
/// flattened into one and the tuned reduce runs over it.
///
/// Which flattening is free depends on the layout the producer left behind. In
/// NCHW `h` and `w` are adjacent and contiguous, so `[b, c, h, w]` reshapes to
/// `[b, c, h * w]` for nothing. A convolution's output is NHWC, where `c` is
/// innermost instead — reshaping *that* to `[b, c, h * w]` materialises a
/// transpose to buy a reshape that is supposed to be free, and the transpose
/// costs more than the reduction it feeds. There the free flattening is
/// `[b, h * w, c]`, reduced over the middle axis, which also leaves `c`
/// innermost and the reduction coalesced across it.
fn global_avg_pool2d(input: CubeTensor) -> CubeTensor {
    let [batch_size, channels, height, width] = input.meta.shape().dims();

    // Channels innermost is the signature of NHWC memory. A single channel is
    // both layouts at once, and the NCHW path is the cheaper one to take.
    let channels_innermost = channels > 1 && input.meta.strides()[1] == 1;

    let (flattened, axis) = match channels_innermost {
        true => (
            reshape(
                permute_nchw_to_nhwc(input),
                Shape::new([batch_size, height * width, channels]),
            ),
            1,
        ),
        false => (
            reshape(input, Shape::new([batch_size, channels, height * width])),
            2,
        ),
    };

    let reduced = reduce_dim(
        flattened,
        None,
        axis,
        KernelReduceStrategy::default(),
        ReduceOperationConfig::Mean,
    )
    .expect("the flattened spatial axis of a rank-3 tensor is reducible");

    reshape(reduced, Shape::new([batch_size, channels, 1, 1]))
}

pub(crate) fn adaptive_avg_pool2d(input: CubeTensor, output_size: [usize; 2]) -> CubeTensor {
    // A `1x1` output is a reduction, and the pooling kernel is the wrong shape
    // of parallelism for it — see [`global_avg_pool2d`].
    if output_size == [1, 1] {
        return global_avg_pool2d(input);
    }

    let [batch_size, channels, _, _] = input.meta.shape().dims();
    let input = into_contiguous_aligned(permute_nchw_to_nhwc(input));

    let output_shape = Shape::new([batch_size, output_size[0], output_size[1], channels]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );

    let mode = PoolMode::from(AdaptiveAvgPoolOptions::new(output_size));

    pool2d(
        &output.client,
        input.clone().binding(),
        output.clone().binding(),
        mode,
        dtype_to_storage_type(output.dtype),
    )
    .unwrap_or_else(|e| pool_panic("adaptive_avg_pool2d", &input, e));

    permute_nhwc_to_nchw(output)
}

pub(crate) fn adaptive_avg_pool2d_backward(x: CubeTensor, out_grad: CubeTensor) -> CubeTensor {
    let [batches, channels, height, width] = x.meta.shape().dims();
    let [_, _, out_h, out_w] = out_grad.meta.shape().dims();
    let input = into_contiguous_aligned(permute_nchw_to_nhwc(x));
    let out_grad = into_contiguous_aligned(permute_nchw_to_nhwc(out_grad));

    let out_shape = Shape::new([batches, height, width, channels]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        out_shape,
        input.dtype,
    );

    let mode = PoolMode::from(AdaptiveAvgPoolOptions::new([out_h, out_w]));

    pool2d_backward(
        &output.client,
        input.clone().binding(),
        out_grad.clone().binding(),
        output.clone().binding(),
        mode,
        dtype_to_storage_type(output.dtype),
    )
    .unwrap_or_else(|e| pool_panic("adaptive_avg_pool2d_backward", &input, e));

    permute_nhwc_to_nchw(output)
}

pub(crate) fn adaptive_avg_pool3d(input: CubeTensor, output_size: [usize; 3]) -> CubeTensor {
    let [batch_size, channels, _, _, _] = input.meta.shape().dims();
    let input = into_contiguous_aligned(permute_nchw_to_nhwc(input));
    let output_shape = Shape::new([
        batch_size,
        output_size[0],
        output_size[1],
        output_size[2],
        channels,
    ]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );
    let mode = PoolMode::from(AdaptiveAvgPoolOptions::new(output_size));

    pool3d(
        &output.client,
        input.clone().binding(),
        output.clone().binding(),
        mode,
        dtype_to_storage_type(output.dtype),
    )
    .unwrap_or_else(|e| pool_panic("adaptive_avg_pool3d", &input, e));

    permute_nhwc_to_nchw(output)
}

pub(crate) fn adaptive_avg_pool3d_backward(x: CubeTensor, out_grad: CubeTensor) -> CubeTensor {
    let [batches, channels, depth, height, width] = x.meta.shape().dims();
    let [_, _, out_depth, out_height, out_width] = out_grad.meta.shape().dims();
    // Cubek only reads the input binding's shape during adaptive average pool 3d backward.
    let input = permute_nchw_to_nhwc(x);
    let out_grad = into_contiguous_aligned(permute_nchw_to_nhwc(out_grad));

    let output_shape = Shape::new([batches, depth, height, width, channels]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );
    let mode = PoolMode::from(AdaptiveAvgPoolOptions::new([
        out_depth, out_height, out_width,
    ]));

    pool3d_backward(
        &output.client,
        input.clone().binding(),
        out_grad.binding(),
        output.clone().binding(),
        mode,
        dtype_to_storage_type(output.dtype),
    )
    .unwrap_or_else(|e| pool_panic("adaptive_avg_pool3d_backward", &input, e));

    permute_nhwc_to_nchw(output)
}

fn pool_panic(label: &str, input: &CubeTensor, error: PoolError) -> ! {
    panic!(
        "{0} kernel failed (device={1:?}, dtype={2:?}): {3}",
        label, input.device, input.dtype, error
    )
}
