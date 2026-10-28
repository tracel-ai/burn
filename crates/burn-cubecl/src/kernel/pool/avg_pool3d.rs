use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{Shape, ops::conv::calculate_pool_output_size};
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::FastDivmod};

use crate::{
    kernel::{
        into_contiguous,
        utils::{address_type, decompose_linear, shape_divmod},
    },
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};

#[derive(CubeLaunch, CubeType)]
struct AvgPool3dArgs {
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_d: usize,
    dilation_h: usize,
    dilation_w: usize,
}

#[cube(launch, address_type = "dynamic")]
fn avg_pool3d_forward_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    out_shape: Sequence<FastDivmod<usize>>,
    args: AvgPool3dArgs,
    #[comptime] count_include_pad: bool,
    #[define(F)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let (_, pos) = decompose_linear(ABSOLUTE_POS, &out_shape);
    let [batch, channel, od, oh, ow] = *pos else {
        unreachable!()
    };

    let in_d_len = input.shape(2);
    let in_h_len = input.shape(3);
    let in_w_len = input.shape(4);

    let stride_d = args.stride_d;
    let stride_h = args.stride_h;
    let stride_w = args.stride_w;

    let pad_d = args.pad_d;
    let pad_h = args.pad_h;
    let pad_w = args.pad_w;

    let kernel_d = args.kernel_d;
    let kernel_h = args.kernel_h;
    let kernel_w = args.kernel_w;

    let dilation_d = args.dilation_d;
    let dilation_h = args.dilation_h;
    let dilation_w = args.dilation_w;

    let mut sum = F::new(0.0);
    let mut count = 0usize;

    let in_base = batch * input.stride(0) + channel * input.stride(1);
    let in_stride_d = input.stride(2);
    let in_stride_h = input.stride(3);
    let in_stride_w = input.stride(4);

    for kd in 0..kernel_d {
        let id_val = od * stride_d + kd * dilation_d;
        if id_val >= pad_d {
            let id = id_val - pad_d;
            if id < in_d_len {
                for kh in 0..kernel_h {
                    let ih_val = oh * stride_h + kh * dilation_h;
                    if ih_val >= pad_h {
                        let ih = ih_val - pad_h;
                        if ih < in_h_len {
                            for kw in 0..kernel_w {
                                let iw_val = ow * stride_w + kw * dilation_w;
                                if iw_val >= pad_w {
                                    let iw = iw_val - pad_w;
                                    if iw < in_w_len {
                                        let in_idx = in_base
                                            + id * in_stride_d
                                            + ih * in_stride_h
                                            + iw * in_stride_w;
                                        sum += input[in_idx];
                                        count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let d_start = od * stride_d;
    let h_start = oh * stride_h;
    let w_start = ow * stride_w;

    let divisor = if comptime!(count_include_pad) {
        padded_taps(d_start, kernel_d, dilation_d, in_d_len + 2 * pad_d)
            * padded_taps(h_start, kernel_h, dilation_h, in_h_len + 2 * pad_h)
            * padded_taps(w_start, kernel_w, dilation_w, in_w_len + 2 * pad_w)
    } else {
        count
    };

    let out_idx = batch * output.stride(0)
        + channel * output.stride(1)
        + od * output.stride(2)
        + oh * output.stride(3)
        + ow * output.stride(4);

    if divisor > 0 {
        output[out_idx] = sum / F::cast_from(divisor);
    } else {
        output[out_idx] = F::new(0.0);
    }
}

/// The number of kernel taps, starting at `start` in the padded input, that land before
/// `padded_len`. Taps past the end padding, from a ceil-mode window, are not counted.
#[cube]
fn padded_taps(start: usize, kernel: usize, dilation: usize, padded_len: usize) -> usize {
    let span = padded_len.saturating_sub(start);
    clamp_max(span.div_ceil(dilation), kernel)
}

/// The number of kernel taps, starting at `start` in the padded input, that land in the
/// input rather than its padding.
#[cube]
fn valid_taps(start: usize, kernel: usize, dilation: usize, pad: usize, in_len: usize) -> usize {
    let first = pad.saturating_sub(start).div_ceil(dilation);
    let end = padded_taps(start, kernel, dilation, in_len + pad);
    end.saturating_sub(first)
}

/// Whether the padded position `pos` is one of the taps of the window that starts at `start`
/// and spans `extent` positions.
#[cube]
fn is_tap(pos: usize, start: usize, extent: usize, dilation: usize) -> bool {
    let mut tap = false;
    if pos >= start {
        let offset = pos - start;
        tap = offset < extent && offset.is_multiple_of(dilation);
    }
    tap
}

#[cube(launch, address_type = "dynamic")]
fn avg_pool3d_backward_kernel<F: Float>(
    grad: &Tensor<F>,
    output_grad: &mut Tensor<F>,
    in_shape: Sequence<FastDivmod<usize>>,
    args: AvgPool3dArgs,
    #[comptime] count_include_pad: bool,
    #[define(F)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= output_grad.len() {
        terminate!();
    }

    let (_, pos) = decompose_linear(ABSOLUTE_POS, &in_shape);
    let [batch, channel, id, ih, iw] = *pos else {
        unreachable!()
    };

    let out_d_len = grad.shape(2);
    let out_h_len = grad.shape(3);
    let out_w_len = grad.shape(4);

    let in_d_len = output_grad.shape(2);
    let in_h_len = output_grad.shape(3);
    let in_w_len = output_grad.shape(4);

    let stride_d = args.stride_d;
    let stride_h = args.stride_h;
    let stride_w = args.stride_w;

    let pad_d = args.pad_d;
    let pad_h = args.pad_h;
    let pad_w = args.pad_w;

    let kernel_d = args.kernel_d;
    let kernel_h = args.kernel_h;
    let kernel_w = args.kernel_w;

    let dilation_d = args.dilation_d;
    let dilation_h = args.dilation_h;
    let dilation_w = args.dilation_w;

    let extent_d = (kernel_d - 1) * dilation_d + 1;
    let extent_h = (kernel_h - 1) * dilation_h + 1;
    let extent_w = (kernel_w - 1) * dilation_w + 1;

    let od_min = if id + pad_d >= extent_d {
        (id + pad_d - extent_d + 1) / stride_d
    } else {
        0usize
    };
    let od_max = clamp_max((id + pad_d) / stride_d + 1, out_d_len);

    let oh_min = if ih + pad_h >= extent_h {
        (ih + pad_h - extent_h + 1) / stride_h
    } else {
        0usize
    };
    let oh_max = clamp_max((ih + pad_h) / stride_h + 1, out_h_len);

    let ow_min = if iw + pad_w >= extent_w {
        (iw + pad_w - extent_w + 1) / stride_w
    } else {
        0usize
    };
    let ow_max = clamp_max((iw + pad_w) / stride_w + 1, out_w_len);

    let grad_base = batch * grad.stride(0) + channel * grad.stride(1);
    let grad_stride_d = grad.stride(2);
    let grad_stride_h = grad.stride(3);
    let grad_stride_w = grad.stride(4);

    let mut sum = F::new(0.0);

    for od in od_min..od_max {
        let d_start = od * stride_d;
        if is_tap(id + pad_d, d_start, extent_d, dilation_d) {
            for oh in oh_min..oh_max {
                let h_start = oh * stride_h;
                if is_tap(ih + pad_h, h_start, extent_h, dilation_h) {
                    for ow in ow_min..ow_max {
                        let w_start = ow * stride_w;
                        if is_tap(iw + pad_w, w_start, extent_w, dilation_w) {
                            let divisor = if comptime!(count_include_pad) {
                                padded_taps(d_start, kernel_d, dilation_d, in_d_len + 2 * pad_d)
                                    * padded_taps(
                                        h_start,
                                        kernel_h,
                                        dilation_h,
                                        in_h_len + 2 * pad_h,
                                    )
                                    * padded_taps(
                                        w_start,
                                        kernel_w,
                                        dilation_w,
                                        in_w_len + 2 * pad_w,
                                    )
                            } else {
                                valid_taps(d_start, kernel_d, dilation_d, pad_d, in_d_len)
                                    * valid_taps(h_start, kernel_h, dilation_h, pad_h, in_h_len)
                                    * valid_taps(w_start, kernel_w, dilation_w, pad_w, in_w_len)
                            };

                            if divisor > 0 {
                                let grad_idx = grad_base
                                    + od * grad_stride_d
                                    + oh * grad_stride_h
                                    + ow * grad_stride_w;
                                sum += grad[grad_idx] / F::cast_from(divisor);
                            }
                        }
                    }
                }
            }
        }
    }

    let out_grad_idx = batch * output_grad.stride(0)
        + channel * output_grad.stride(1)
        + id * output_grad.stride(2)
        + ih * output_grad.stride(3)
        + iw * output_grad.stride(4);

    output_grad[out_grad_idx] = sum;
}

pub fn avg_pool3d(
    x: CubeTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    count_include_pad: bool,
    ceil_mode: bool,
) -> CubeTensor {
    let [batch_size, channels, in_d, in_h, in_w] = x.meta.shape().dims();

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        in_d,
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        in_h,
        ceil_mode,
    );
    let size_2 = calculate_pool_output_size(
        kernel_size[2],
        stride[2],
        padding[2],
        dilation[2],
        in_w,
        ceil_mode,
    );

    let x = into_contiguous(x);
    let shape_out = Shape::new([batch_size, channels, size_0, size_1, size_2]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, x.dtype);

    let num_elems = output.meta.num_elements();
    let cube_dim = CubeDim::new(&x.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&x.client, num_elems, cube_dim);

    let dtype = x.dtype;

    avg_pool3d_forward_kernel::launch(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(x, output),
        x.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        shape_divmod(&output),
        AvgPool3dArgsLaunch::new(
            stride[0],
            stride[1],
            stride[2],
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            padding[0],
            padding[1],
            padding[2],
            dilation[0],
            dilation[1],
            dilation[2],
        ),
        count_include_pad,
        dtype_to_storage_type(dtype),
    );

    output
}

#[allow(clippy::too_many_arguments)]
pub fn avg_pool3d_backward(
    x: CubeTensor,
    grad: CubeTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    count_include_pad: bool,
    _ceil_mode: bool,
) -> CubeTensor {
    let x = into_contiguous(x);
    let grad = into_contiguous(grad);

    let output_grad = empty_device_dtype(
        x.client.clone(),
        x.device.clone(),
        x.meta.shape().clone(),
        x.dtype,
    );

    let num_elems = output_grad.meta.num_elements();
    let cube_dim = CubeDim::new(&x.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&x.client, num_elems, cube_dim);

    let dtype = x.dtype;

    avg_pool3d_backward_kernel::launch(
        &output_grad.client,
        cube_count,
        cube_dim,
        address_type!(grad, output_grad),
        grad.into_tensor_arg(),
        output_grad.clone().into_tensor_arg(),
        shape_divmod(&output_grad),
        AvgPool3dArgsLaunch::new(
            stride[0],
            stride[1],
            stride[2],
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            padding[0],
            padding[1],
            padding[2],
            dilation[0],
            dilation[1],
            dilation[2],
        ),
        count_include_pad,
        dtype_to_storage_type(dtype),
    );

    output_grad
}
