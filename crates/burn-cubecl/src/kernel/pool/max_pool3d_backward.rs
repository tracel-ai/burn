use crate::{
    CubeRuntime,
    kernel::into_contiguous,
    ops::{
        max_line_size, numeric::empty_device_dtype, permute_ncdhw_to_ndhwc, permute_ndhwc_to_ncdhw,
    },
    tensor::CubeTensor,
};
use burn_backend::Shape;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Pool3dBackwardArgs {
    pub stride_0: i32,
    pub stride_1: i32,
    pub stride_2: i32,
    pub dilation_0: i32,
    pub dilation_1: i32,
    pub dilation_2: i32,
    pub padding_0: i32,
    pub padding_1: i32,
    pub padding_2: i32,
}

#[cube(launch_unchecked)]
fn max_pool3d_with_indices_backward_kernel<E: Numeric, I: Int>(
    grad: &Tensor<Line<E>>,
    indices: &Tensor<Line<I>>,
    output: &mut Tensor<Line<E>>,
    args: &Pool3dBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
    #[comptime] kernel_size_2: i32,
    #[define(E, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let line_size = grad.line_size();

    // Output shape is [batch, depth, height, width, channels] in NDHWC format
    let channels = output.shape(4) / line_size;
    let channel = (ABSOLUTE_POS % channels) * output.line_size();
    let pos = ABSOLUTE_POS / channels;
    let iw = pos % output.shape(3);
    let pos = pos / output.shape(3);
    let ih = pos % output.shape(2);
    let pos = pos / output.shape(2);
    let id = pos % output.shape(1);
    let batch = pos / output.shape(1);

    let index_current = id * output.shape(2) * output.shape(3) + ih * output.shape(3) + iw;

    let (od_start, od_end, oh_start, oh_end, ow_start, ow_end) = loop_ranges(
        id as i32,
        ih as i32,
        iw as i32,
        grad.shape(1) as u32,
        grad.shape(2) as u32,
        grad.shape(3) as u32,
        args,
        kernel_size_0,
        kernel_size_1,
        kernel_size_2,
    );

    let mut grad_acc = Line::empty(grad.line_size()).fill(E::from_int(0));

    let index_base = batch * grad.stride(0) + channel * grad.stride(4);

    for od in od_start..od_end {
        for oh in oh_start..oh_end {
            for ow in ow_start..ow_end {
                let index = index_base
                    + od as usize * grad.stride(1)
                    + oh as usize * grad.stride(2)
                    + ow as usize * grad.stride(3);
                let index_max = Line::<u32>::cast_from(indices[index / line_size]);

                grad_acc += select_many(
                    index_max.equal(Line::cast_from(index_current)),
                    grad[index / line_size],
                    Line::new(E::from_int(0)),
                );
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn max_pool3d_with_indices_backward<R: CubeRuntime>(
    x: CubeTensor<R>,
    grad: CubeTensor<R>,
    indices: CubeTensor<R>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    _ceil_mode: bool,
) -> CubeTensor<R> {
    let [batches, channels, depth, height, width] = x.shape.dims();

    let grad = into_contiguous(permute_ncdhw_to_ndhwc(grad));
    let indices = into_contiguous(permute_ncdhw_to_ndhwc(indices));

    let line_size = if grad.strides[4] == indices.strides[4] {
        max_line_size(&grad)
    } else {
        1
    };

    let out_shape = Shape::new([batches, depth, height, width, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), out_shape, x.dtype);

    let working_units = output.shape.num_elements() / line_size as usize;
    let cube_dim = CubeDim::new(&x.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&x.client, working_units, cube_dim);

    unsafe {
        max_pool3d_with_indices_backward_kernel::launch_unchecked(
            &x.client,
            cube_count,
            cube_dim,
            grad.as_tensor_arg(line_size),
            indices.as_tensor_arg(line_size),
            output.as_tensor_arg(line_size),
            Pool3dBackwardArgsLaunch::new(
                ScalarArg::new(stride[0] as i32),
                ScalarArg::new(stride[1] as i32),
                ScalarArg::new(stride[2] as i32),
                ScalarArg::new(dilation[0] as i32),
                ScalarArg::new(dilation[1] as i32),
                ScalarArg::new(dilation[2] as i32),
                ScalarArg::new(padding[0] as i32),
                ScalarArg::new(padding[1] as i32),
                ScalarArg::new(padding[2] as i32),
            ),
            kernel_size[0] as i32,
            kernel_size[1] as i32,
            kernel_size[2] as i32,
            [x.dtype.into(), indices.dtype.into()],
        )
        .expect("Kernel to never fail")
    };

    permute_ndhwc_to_ncdhw(output)
}
