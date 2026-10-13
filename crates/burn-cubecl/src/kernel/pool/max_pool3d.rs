use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, Shape, ops::conv::calculate_pool_output_size};
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
struct MaxPool3dArgs {
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
fn max_pool3d_forward_kernel<F: Float, I: Int>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    indices: &mut Tensor<I>,
    out_shape: Sequence<FastDivmod<usize>>,
    args: MaxPool3dArgs,
    #[define(F, I)] _dtypes: [ElemType; 2],
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

    let mut max_val = F::new(f32::NEG_INFINITY);
    let mut max_idx = 0usize;
    let mut first_valid = true;

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
                                        let val = input[in_idx];
                                        let flat_idx =
                                            id * (in_h_len * in_w_len) + ih * in_w_len + iw;
                                        if first_valid || val > max_val {
                                            max_val = val;
                                            max_idx = flat_idx;
                                            first_valid = false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let out_idx = batch * output.stride(0)
        + channel * output.stride(1)
        + od * output.stride(2)
        + oh * output.stride(3)
        + ow * output.stride(4);

    output[out_idx] = max_val;
    if first_valid {
        indices[out_idx] = I::cast_from(-1i32);
    } else {
        indices[out_idx] = I::cast_from(max_idx);
    }
}

#[cube(launch, address_type = "dynamic")]
fn max_pool3d_backward_kernel<F: Float, I: Int>(
    grad: &Tensor<F>,
    indices: &Tensor<I>,
    output_grad: &mut Tensor<F>,
    in_shape: Sequence<FastDivmod<usize>>,
    args: MaxPool3dArgs,
    #[define(F, I)] _dtypes: [ElemType; 2],
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

    let effective_k_d = if kernel_d > 0 {
        (kernel_d - 1) * dilation_d
    } else {
        0usize
    };
    let effective_k_h = if kernel_h > 0 {
        (kernel_h - 1) * dilation_h
    } else {
        0usize
    };
    let effective_k_w = if kernel_w > 0 {
        (kernel_w - 1) * dilation_w
    } else {
        0usize
    };

    let od_min = if id + pad_d >= effective_k_d {
        (id + pad_d - effective_k_d) / stride_d
    } else {
        0usize
    };
    let od_max = clamp_max((id + pad_d) / stride_d + 1, out_d_len);

    let oh_min = if ih + pad_h >= effective_k_h {
        (ih + pad_h - effective_k_h) / stride_h
    } else {
        0usize
    };
    let oh_max = clamp_max((ih + pad_h) / stride_h + 1, out_h_len);

    let ow_min = if iw + pad_w >= effective_k_w {
        (iw + pad_w - effective_k_w) / stride_w
    } else {
        0usize
    };
    let ow_max = clamp_max((iw + pad_w) / stride_w + 1, out_w_len);

    let my_idx = I::cast_from(id * (in_h_len * in_w_len) + ih * in_w_len + iw);

    let mut sum = F::new(0.0_f32);
    let grad_base = batch * grad.stride(0) + channel * grad.stride(1);
    let idx_base = batch * indices.stride(0) + channel * indices.stride(1);

    for od in od_min..od_max {
        for oh in oh_min..oh_max {
            for ow in ow_min..ow_max {
                let idx_offset = idx_base
                    + od * indices.stride(2)
                    + oh * indices.stride(3)
                    + ow * indices.stride(4);
                if indices[idx_offset] == my_idx {
                    let grad_offset =
                        grad_base + od * grad.stride(2) + oh * grad.stride(3) + ow * grad.stride(4);
                    sum += grad[grad_offset];
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

pub fn max_pool3d_with_indices(
    x: CubeTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
    dtype_indices: DType,
) -> (CubeTensor, CubeTensor) {
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
    let output = empty_device_dtype(
        x.client.clone(),
        x.device.clone(),
        shape_out.clone(),
        x.dtype,
    );
    let indices = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, dtype_indices);

    let num_elems = output.meta.num_elements();
    let cube_dim = CubeDim::new(&x.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&x.client, num_elems, cube_dim);

    let dtype = x.dtype;

    max_pool3d_forward_kernel::launch(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(x, output, indices),
        x.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        indices.clone().into_tensor_arg(),
        shape_divmod(&output),
        MaxPool3dArgsLaunch::new(
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
        [
            dtype_to_storage_type(dtype),
            dtype_to_storage_type(dtype_indices),
        ],
    );

    (output, indices)
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool3d_with_indices_backward(
    x: CubeTensor,
    grad: CubeTensor,
    indices: CubeTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    _ceil_mode: bool,
) -> CubeTensor {
    let x = into_contiguous(x);
    let grad = into_contiguous(grad);
    let indices = into_contiguous(indices);

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
    let indices_dtype = indices.dtype;

    max_pool3d_backward_kernel::launch(
        &output_grad.client,
        cube_count,
        cube_dim,
        address_type!(grad, indices, output_grad),
        grad.into_tensor_arg(),
        indices.into_tensor_arg(),
        output_grad.clone().into_tensor_arg(),
        shape_divmod(&output_grad),
        MaxPool3dArgsLaunch::new(
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
        [
            dtype_to_storage_type(dtype),
            dtype_to_storage_type(indices_dtype),
        ],
    );

    output_grad
}
