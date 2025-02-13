use crate::{
    element::CubeElement, kernel::into_contiguous, ops::numeric::empty_device, tensor::CubeTensor,
    CubeRuntime,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[derive(CubeLaunch)]
pub(crate) struct PoolBackwardArgs {
    pub stride_0: i32,
    pub stride_1: i32,
    pub dilation_0: i32,
    pub dilation_1: i32,
    pub padding_0: i32,
    pub padding_1: i32,
}

#[cube(launch_unchecked)]
fn avg_pool2d_backward_kernel<E: Numeric>(
    grad: &Tensor<E>,
    output: &mut Tensor<E>,
    args: &PoolBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
    #[comptime] count_include_pad: bool,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let batch = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let channel = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let ih = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let iw = ABSOLUTE_POS / output.stride(3) % output.shape(3);

    let mut grad_acc = E::from_int(0);

    let (oh_start, oh_end, ow_start, ow_end) = loop_ranges(
        ih as i32,
        iw as i32,
        grad.shape(2),
        grad.shape(3),
        args,
        kernel_size_0,
        kernel_size_1,
    );

    let padding_0 = args.padding_0 as u32;
    let padding_1 = args.padding_1 as u32;
    let stride_0 = args.stride_0 as u32;
    let stride_1 = args.stride_1 as u32;
    let kernel_size_0 = comptime![kernel_size_0 as u32];
    let kernel_size_1 = comptime![kernel_size_1 as u32];

    let index_base = batch * grad.stride(0) + channel * grad.stride(1);
    let border_bottom = output.shape(2) + padding_0;
    let border_right = output.shape(3) + padding_1;
    let begin_h = ih + padding_0;
    let begin_w = iw + padding_1;

    for oh in oh_start..oh_end {
        let ih_start = oh * stride_0;
        let ih_end = Min::min(ih_start + kernel_size_0, border_bottom);
        let ih_start = Max::max(ih_start, padding_0);

        if begin_h >= ih_start && ih < ih_end {
            for ow in ow_start..ow_end {
                let index = index_base + oh * grad.stride(2) + ow * grad.stride(3);

                let iw_start = ow * stride_1;
                let iw_end = Min::min(iw_start + kernel_size_1, border_right);
                let iw_start = Max::max(iw_start, padding_1);

                if begin_w >= iw_start && iw < iw_end {
                    if count_include_pad {
                        grad_acc += grad[index] / E::cast_from(kernel_size_0 * kernel_size_1);
                    } else {
                        let ih_diff = ih_end - ih_start;
                        let iw_diff = iw_end - iw_start;
                        let count = E::cast_from(ih_diff * iw_diff);
                        grad_acc += grad[index] / count;
                    }
                }
            }
        }
    }

    output[ABSOLUTE_POS] = grad_acc;
}

#[cube]
fn loop_ranges(
    ih: i32,
    iw: i32,
    grad_h: u32,
    grad_w: u32,
    args: &PoolBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
) -> (u32, u32, u32, u32) {
    let kms_0 = args.dilation_0 * kernel_size_0 - args.stride_0;
    let kms_1 = args.dilation_1 * kernel_size_1 - args.stride_1;

    let oh_start = Max::max((ih + args.padding_0 - kms_0) / args.stride_0, 0) as u32;
    let ow_start = Max::max((iw + args.padding_1 - kms_1) / args.stride_1, 0) as u32;
    let oh_end = Min::min(Max::max(kms_0, 0) as u32 + oh_start, grad_h - 1) + 1;
    let ow_end = Min::min(Max::max(kms_1, 0) as u32 + ow_start, grad_w - 1) + 1;

    (oh_start, oh_end, ow_start, ow_end)
}

pub(crate) fn avg_pool2d_backward<R: CubeRuntime, E: CubeElement>(
    x: CubeTensor<R>,
    grad: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> CubeTensor<R> {
    let grad = into_contiguous(grad);
    let dilation = 1;

    let output = empty_device::<R, E>(x.client.clone(), x.device.clone(), x.shape.clone());
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    unsafe {
        avg_pool2d_backward_kernel::launch_unchecked::<E, R>(
            &grad.client,
            cube_count,
            cube_dim,
            grad.as_tensor_arg::<E>(1),
            output.as_tensor_arg::<E>(1),
            PoolBackwardArgsLaunch::new(
                ScalarArg::new(stride[0] as i32),
                ScalarArg::new(stride[1] as i32),
                ScalarArg::new(dilation),
                ScalarArg::new(dilation),
                ScalarArg::new(padding[0] as i32),
                ScalarArg::new(padding[1] as i32),
            ),
            kernel_size[0] as i32,
            kernel_size[1] as i32,
            count_include_pad,
        )
    };

    output
}
