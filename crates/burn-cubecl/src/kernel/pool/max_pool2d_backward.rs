use crate::{
    element::CubeElement, kernel::into_contiguous, ops::numeric::empty_device, tensor::CubeTensor,
    CubeRuntime, IntElement,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use super::{PoolBackwardArgs, PoolBackwardArgsLaunch};

#[cube(launch_unchecked)]
fn max_pool2d_with_indices_backward_kernel<E: Numeric, I: Int>(
    grad: &Tensor<E>,
    indices: &Tensor<I>,
    output: &mut Tensor<E>,
    args: &PoolBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let batch = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let channel = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let ih = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let iw = ABSOLUTE_POS / output.stride(3) % output.shape(3);

    let index_current = ih * output.stride(2) + iw * output.stride(3);

    let (oh_start, oh_end, ow_start, ow_end) = loop_ranges(
        ih as i32,
        iw as i32,
        grad.shape(2),
        grad.shape(3),
        args,
        kernel_size_0,
        kernel_size_1,
    );

    let mut grad_acc = E::from_int(0);

    let index_base = batch * grad.stride(0) + channel * grad.stride(1);

    for oh in oh_start..oh_end {
        for ow in ow_start..ow_end {
            let index = index_base + oh * grad.stride(2) + ow * grad.stride(3);
            let index_max = u32::cast_from(indices[index]);

            grad_acc += select(index_max == index_current, grad[index], E::from_int(0));
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

pub(crate) fn max_pool2d_with_indices_backward<R: CubeRuntime, E: CubeElement, I: IntElement>(
    x: CubeTensor<R>,
    grad: CubeTensor<R>,
    indices: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> CubeTensor<R> {
    let grad = into_contiguous(grad);
    let indices = into_contiguous(indices);

    let output = empty_device::<R, E>(x.client.clone(), x.device.clone(), x.shape.clone());
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    unsafe {
        max_pool2d_with_indices_backward_kernel::launch_unchecked::<E, I, R>(
            &x.client,
            cube_count,
            cube_dim,
            grad.as_tensor_arg::<E>(1),
            indices.as_tensor_arg::<I>(1),
            output.as_tensor_arg::<E>(1),
            PoolBackwardArgsLaunch::new(
                ScalarArg::new(stride[0] as i32),
                ScalarArg::new(stride[1] as i32),
                ScalarArg::new(dilation[0] as i32),
                ScalarArg::new(dilation[0] as i32),
                ScalarArg::new(padding[0] as i32),
                ScalarArg::new(padding[1] as i32),
            ),
            kernel_size[0] as i32,
            kernel_size[1] as i32,
        )
    };

    output
}
