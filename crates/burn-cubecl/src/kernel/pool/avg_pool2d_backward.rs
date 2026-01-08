use crate::{
    CubeRuntime,
    ops::{max_line_size, numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_backend::Shape;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[derive(CubeLaunch, CubeType)]
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
    grad: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    args: &PoolBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
    #[comptime] count_include_pad: bool,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let line_size = grad.line_size();

    let channel_lines = output.shape(3) / line_size;
    let channel = (ABSOLUTE_POS % channel_lines) * output.line_size();
    let pos = ABSOLUTE_POS / channel_lines;
    let iw = pos as u32 % output.shape(2) as u32;
    let pos = pos / output.shape(2);
    let ih = pos as u32 % output.shape(1) as u32;
    let batch = pos / output.shape(1);

    let mut grad_acc = Line::empty(grad.line_size()).fill(E::from_int(0));

    let (oh_start, oh_end, ow_start, ow_end) = loop_ranges(
        ih as i32,
        iw as i32,
        grad.shape(1) as u32,
        grad.shape(2) as u32,
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

    let index_base = batch * grad.stride(0) + channel * grad.stride(3);
    let border_bottom = output.shape(1) as u32 + padding_0;
    let border_right = output.shape(2) as u32 + padding_1;
    let begin_h = ih + padding_0;
    let begin_w = iw + padding_1;

    for oh in oh_start..oh_end {
        let ih_start = oh * stride_0;
        let ih_end = clamp_max(ih_start + kernel_size_0, border_bottom);
        let ih_start = clamp_min(ih_start, padding_0);

        if begin_h >= ih_start && ih < ih_end {
            for ow in ow_start..ow_end {
                let index =
                    index_base + oh as usize * grad.stride(1) + ow as usize * grad.stride(2);

                let iw_start = ow * stride_1;
                let iw_end = clamp_max(iw_start + kernel_size_1, border_right);
                let iw_start = clamp_min(iw_start, padding_1);

                if begin_w >= iw_start && iw < iw_end {
                    if count_include_pad {
                        grad_acc += grad[index / line_size]
                            / Line::cast_from(kernel_size_0 * kernel_size_1);
                    } else {
                        let ih_diff = ih_end - ih_start;
                        let iw_diff = iw_end - iw_start;
                        let count = Line::cast_from(ih_diff * iw_diff);
                        grad_acc += grad[index / line_size] / count;
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

    let oh_start = clamp_min((ih + args.padding_0 - kms_0) / args.stride_0, 0) as u32;
    let ow_start = clamp_min((iw + args.padding_1 - kms_1) / args.stride_1, 0) as u32;
    let oh_end = clamp_max(clamp_min(kms_0, 0) as u32 + oh_start, grad_h - 1) + 1;
    let ow_end = clamp_max(clamp_min(kms_1, 0) as u32 + ow_start, grad_w - 1) + 1;

    (oh_start, oh_end, ow_start, ow_end)
}

pub(crate) fn avg_pool2d_backward<R: CubeRuntime>(
    x: CubeTensor<R>,
    grad: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    _ceil_mode: bool,
) -> CubeTensor<R> {
    let [batches, channels, height, width] = x.shape.dims();

    let grad = permute_nchw_to_nhwc(grad);

    let line_size = if x.strides[3] == grad.strides[3] {
        max_line_size(&x)
    } else {
        1
    };

    let dilation = 1;

    let out_shape = Shape::new([batches, height, width, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), out_shape, x.dtype);

    let working_units = output.shape.num_elements() / line_size as usize;
    let cube_dim = CubeDim::new(&x.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&x.client, working_units, cube_dim);

    unsafe {
        avg_pool2d_backward_kernel::launch_unchecked(
            &grad.client,
            cube_count,
            cube_dim,
            grad.as_tensor_arg(line_size),
            output.as_tensor_arg(line_size),
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
            output.dtype.into(),
        )
    }
    .expect("Kernel to never fail");

    permute_nhwc_to_nchw(output)
}
