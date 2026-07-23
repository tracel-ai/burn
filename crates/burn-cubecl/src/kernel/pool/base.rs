use crate::{
    CubeRuntime,
    kernel::into_contiguous_aligned,
    ops::{
        base::{from_data, into_data_sync},
        numeric::empty_device_dtype,
        permute_nchw_to_nhwc, permute_nhwc_to_nchw,
    },
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, Shape, TensorData, ops::conv::calculate_pool_output_size};
use cubek::pool::{
    definition::{AdaptiveAvgPoolOptions, AvgPoolOptions, MaxPoolOptions, PoolError, PoolMode},
    pool2d, pool2d_backward, pool2d_with_indices, pool2d_with_indices_backward,
};

pub(crate) fn max_pool2d<R: CubeRuntime>(
    x: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> CubeTensor<R> {
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

pub(crate) fn max_pool2d_with_indices<R: CubeRuntime>(
    x: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    dtype_indices: DType,
) -> (CubeTensor<R>, CubeTensor<R>) {
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
pub(crate) fn max_pool2d_with_indices_backward<R: CubeRuntime>(
    x: CubeTensor<R>,
    grad: CubeTensor<R>,
    indices: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> CubeTensor<R> {
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

pub(crate) fn avg_pool2d<R: CubeRuntime>(
    x: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> CubeTensor<R> {
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

pub(crate) fn avg_pool2d_backward<R: CubeRuntime>(
    x: CubeTensor<R>,
    grad: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> CubeTensor<R> {
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

pub(crate) fn adaptive_avg_pool2d<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
) -> CubeTensor<R> {
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

pub(crate) fn adaptive_avg_pool2d_backward<R: CubeRuntime>(
    x: CubeTensor<R>,
    out_grad: CubeTensor<R>,
) -> CubeTensor<R> {
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

pub(crate) fn adaptive_avg_pool3d<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 3],
) -> CubeTensor<R> {
    // ponytail: CPU fallback until cubek adds pool3d support
    let [batch_size, channels, input_depth, input_height, input_width] = input.meta.shape().dims();

    let input_data = into_data_sync(input.clone());
    let input_slice = input_data.as_slice::<f32>().unwrap();

    let numel = batch_size * channels * output_size[0] * output_size[1] * output_size[2];
    let mut output_data = vec![0.0f32; numel];

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..output_size[0] {
                for oh in 0..output_size[1] {
                    for ow in 0..output_size[2] {
                        let id_start = start_index(od, output_size[0], input_depth);
                        let id_end = end_index(od, output_size[0], input_depth);
                        let ih_start = start_index(oh, output_size[1], input_height);
                        let ih_end = end_index(oh, output_size[1], input_height);
                        let iw_start = start_index(ow, output_size[2], input_width);
                        let iw_end = end_index(ow, output_size[2], input_width);

                        let mut sum = 0.0f32;
                        for id in id_start..id_end {
                            for ih in ih_start..ih_end {
                                for iw in iw_start..iw_end {
                                    let idx =
                                        b * channels * input_depth * input_height * input_width
                                            + c * input_depth * input_height * input_width
                                            + id * input_height * input_width
                                            + ih * input_width
                                            + iw;
                                    sum += input_slice[idx];
                                }
                            }
                        }

                        let count = ((id_end - id_start)
                            * (ih_end - ih_start)
                            * (iw_end - iw_start)) as f32;
                        let out_idx =
                            b * channels * output_size[0] * output_size[1] * output_size[2]
                                + c * output_size[0] * output_size[1] * output_size[2]
                                + od * output_size[1] * output_size[2]
                                + oh * output_size[2]
                                + ow;
                        output_data[out_idx] = sum / count;
                    }
                }
            }
        }
    }

    let output_shape = Shape::new([
        batch_size,
        channels,
        output_size[0],
        output_size[1],
        output_size[2],
    ]);
    let output_tensor = TensorData::new(output_data, output_shape);
    from_data(output_tensor, &input.device)
}

pub(crate) fn adaptive_avg_pool3d_backward<R: CubeRuntime>(
    x: CubeTensor<R>,
    out_grad: CubeTensor<R>,
) -> CubeTensor<R> {
    let [_, _, input_depth, input_height, input_width] = x.meta.shape().dims();
    let [
        batch_size,
        channels,
        output_depth,
        output_height,
        output_width,
    ] = out_grad.meta.shape().dims();

    let grad_data = into_data_sync(out_grad);
    let grad_slice = grad_data.as_slice::<f32>().unwrap();

    let numel = batch_size * channels * input_depth * input_height * input_width;
    let mut output_data = vec![0.0f32; numel];

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..output_depth {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let id_start = start_index(od, output_depth, input_depth);
                        let id_end = end_index(od, output_depth, input_depth);
                        let ih_start = start_index(oh, output_height, input_height);
                        let ih_end = end_index(oh, output_height, input_height);
                        let iw_start = start_index(ow, output_width, input_width);
                        let iw_end = end_index(ow, output_width, input_width);

                        let count = ((id_end - id_start)
                            * (ih_end - ih_start)
                            * (iw_end - iw_start)) as f32;
                        let grad_idx = b * channels * output_depth * output_height * output_width
                            + c * output_depth * output_height * output_width
                            + od * output_height * output_width
                            + oh * output_width
                            + ow;
                        let grad_val = grad_slice[grad_idx] / count;

                        for id in id_start..id_end {
                            for ih in ih_start..ih_end {
                                for iw in iw_start..iw_end {
                                    let idx =
                                        b * channels * input_depth * input_height * input_width
                                            + c * input_depth * input_height * input_width
                                            + id * input_height * input_width
                                            + ih * input_width
                                            + iw;
                                    output_data[idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let out_shape = Shape::new([batch_size, channels, input_depth, input_height, input_width]);
    let output_tensor = TensorData::new(output_data, out_shape);
    from_data(output_tensor, &x.device)
}

fn start_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    ((output_size_index as f32 * input_size as f32) / output_size as f32).floor() as usize
}

fn end_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    let index =
        (((output_size_index + 1) as f32 * input_size as f32) / output_size as f32).ceil() as usize;
    usize::min(index, input_size)
}

fn pool_panic<R: CubeRuntime>(label: &str, input: &CubeTensor<R>, error: PoolError) -> ! {
    panic!(
        "{0} kernel failed (device={1:?}, dtype={2:?}): {3}",
        label, input.device, input.dtype, error
    )
}
