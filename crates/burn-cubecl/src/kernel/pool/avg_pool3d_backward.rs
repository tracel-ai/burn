use crate::{
    CubeRuntime,
    ops::{
        max_line_size, numeric::empty_device_dtype, permute_ncdhw_to_ndhwc, permute_ndhwc_to_ncdhw,
    },
    tensor::CubeTensor,
};
use burn_backend::Shape;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use super::max_pool3d_backward::{Pool3dBackwardArgs, Pool3dBackwardArgsLaunch};

#[cube(launch_unchecked)]
fn avg_pool3d_backward_kernel<E: Numeric>(
    grad: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    args: &Pool3dBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
    #[comptime] kernel_size_2: i32,
    #[comptime] count_include_pad: bool,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let line_size = grad.line_size();

    // Output shape is [batch, depth, height, width, channels] in NDHWC format
    let channel_lines = output.shape(4) / line_size;
    let channel = (ABSOLUTE_POS % channel_lines) * output.line_size();
    let pos = ABSOLUTE_POS / channel_lines;
    let iw = pos % output.shape(3);
    let pos = pos / output.shape(3);
    let ih = pos % output.shape(2);
    let pos = pos / output.shape(2);
    let id = pos % output.shape(1);
    let batch = pos / output.shape(1);

    let mut grad_acc = Line::empty(grad.line_size()).fill(E::from_int(0));

    let (od_start, od_end, oh_start, oh_end, ow_start, ow_end) = loop_ranges(
        id as i32,
        ih as i32,
        iw as i32,
        grad.shape(1),
        grad.shape(2),
        grad.shape(3),
        args,
        kernel_size_0,
        kernel_size_1,
        kernel_size_2,
    );

    let padding_0 = args.padding_0 as u32;
    let padding_1 = args.padding_1 as u32;
    let padding_2 = args.padding_2 as u32;
    let stride_0 = args.stride_0 as u32;
    let stride_1 = args.stride_1 as u32;
    let stride_2 = args.stride_2 as u32;
    let kernel_size_0 = comptime![kernel_size_0 as u32];
    let kernel_size_1 = comptime![kernel_size_1 as u32];
    let kernel_size_2 = comptime![kernel_size_2 as u32];

    let index_base = batch * grad.stride(0) + channel * grad.stride(4);
    let border_back = output.shape(1) + padding_0;
    let border_bottom = output.shape(2) + padding_1;
    let border_right = output.shape(3) + padding_2;
    let begin_d = id + padding_0;
    let begin_h = ih + padding_1;
    let begin_w = iw + padding_2;

    for od in od_start..od_end {
        let id_start = od * stride_0;
        let id_end = Min::min(id_start + kernel_size_0, border_back);
        let id_start = Max::max(id_start, padding_0);

        if begin_d >= id_start && id < id_end {
            for oh in oh_start..oh_end {
                let ih_start = oh * stride_1;
                let ih_end = Min::min(ih_start + kernel_size_1, border_bottom);
                let ih_start = Max::max(ih_start, padding_1);

                if begin_h >= ih_start && ih < ih_end {
                    for ow in ow_start..ow_end {
                        let index = index_base
                            + od * grad.stride(1)
                            + oh * grad.stride(2)
                            + ow * grad.stride(3);

                        let iw_start = ow * stride_2;
                        let iw_end = Min::min(iw_start + kernel_size_2, border_right);
                        let iw_start = Max::max(iw_start, padding_2);

                        if begin_w >= iw_start && iw < iw_end {
                            if count_include_pad {
                                grad_acc += grad[index / line_size]
                                    / Line::cast_from(
                                        kernel_size_0 * kernel_size_1 * kernel_size_2,
                                    );
                            } else {
                                let id_diff = id_end - id_start;
                                let ih_diff = ih_end - ih_start;
                                let iw_diff = iw_end - iw_start;
                                let count = Line::cast_from(id_diff * ih_diff * iw_diff);
                                grad_acc += grad[index / line_size] / count;
                            }
                        }
                    }
                }
            }
        }
    }

    output[ABSOLUTE_POS] = grad_acc;
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn loop_ranges(
    id: i32,
    ih: i32,
    iw: i32,
    grad_d: u32,
    grad_h: u32,
    grad_w: u32,
    args: &Pool3dBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
    #[comptime] kernel_size_2: i32,
) -> (u32, u32, u32, u32, u32, u32) {
    let kms_0 = args.dilation_0 * kernel_size_0 - args.stride_0;
    let kms_1 = args.dilation_1 * kernel_size_1 - args.stride_1;
    let kms_2 = args.dilation_2 * kernel_size_2 - args.stride_2;

    let od_start = Max::max((id + args.padding_0 - kms_0) / args.stride_0, 0) as u32;
    let oh_start = Max::max((ih + args.padding_1 - kms_1) / args.stride_1, 0) as u32;
    let ow_start = Max::max((iw + args.padding_2 - kms_2) / args.stride_2, 0) as u32;

    let od_end = Min::min(Max::max(kms_0, 0) as u32 + od_start, grad_d - 1) + 1;
    let oh_end = Min::min(Max::max(kms_1, 0) as u32 + oh_start, grad_h - 1) + 1;
    let ow_end = Min::min(Max::max(kms_2, 0) as u32 + ow_start, grad_w - 1) + 1;

    (od_start, od_end, oh_start, oh_end, ow_start, ow_end)
}

pub(crate) fn avg_pool3d_backward<R: CubeRuntime>(
    x: CubeTensor<R>,
    grad: CubeTensor<R>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    count_include_pad: bool,
    _ceil_mode: bool,
) -> CubeTensor<R> {
    let [batches, channels, depth, height, width] = x.shape.dims();

    let grad = permute_ncdhw_to_ndhwc(grad);

    let line_size = if x.strides[4] == grad.strides[4] {
        max_line_size(&x)
    } else {
        1
    };

    let dilation = 1;

    let out_shape = Shape::new([batches, depth, height, width, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), out_shape, x.dtype);

    let working_units = output.shape.num_elements() / line_size as usize;
    let cube_dim = CubeDim::new(&x.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&x.client, working_units, cube_dim);

    unsafe {
        avg_pool3d_backward_kernel::launch_unchecked(
            &grad.client,
            cube_count,
            cube_dim,
            grad.as_tensor_arg(line_size),
            output.as_tensor_arg(line_size),
            Pool3dBackwardArgsLaunch::new(
                ScalarArg::new(stride[0] as i32),
                ScalarArg::new(stride[1] as i32),
                ScalarArg::new(stride[2] as i32),
                ScalarArg::new(dilation),
                ScalarArg::new(dilation),
                ScalarArg::new(dilation),
                ScalarArg::new(padding[0] as i32),
                ScalarArg::new(padding[1] as i32),
                ScalarArg::new(padding[2] as i32),
            ),
            kernel_size[0] as i32,
            kernel_size[1] as i32,
            kernel_size[2] as i32,
            count_include_pad,
            output.dtype.into(),
        )
    }
    .expect("Kernel to never fail");

    permute_ndhwc_to_ncdhw(output)
}
