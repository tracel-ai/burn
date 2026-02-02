use crate::{
    CubeRuntime,
    kernel::{
        into_contiguous_aligned,
        pool::pool2d::{Position, view4d},
        utils::{decompose_linear, shape_divmod},
    },
    ops::{max_line_size, numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_backend::Shape;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, tensor::View},
};

#[cube(launch)]
fn adaptive_avg_pool2d_backward_direct<E: Numeric>(
    grad: &Tensor<Line<E>>,
    output: &mut View<Line<E>, Position, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (_, out_h, out_w, _) = output.shape();
    let (grad_stride_h, grad_stride_w) = (grad.stride(1), grad.stride(2));
    let (grad_h, grad_w) = (grad.shape(1), grad.shape(2));

    let (_, pos) = decompose_linear(ABSOLUTE_POS * output.line_size(), &out_shape);
    let [b, ih, iw, c] = *pos else { unreachable!() };

    let oh_start = start_index(ih, out_h, grad_h);
    let oh_end = end_index(ih, out_h, grad_h);

    let ow_start = start_index(iw, out_w, grad_w);
    let ow_end = end_index(iw, out_w, grad_w);

    let mut grad_acc = Line::empty(grad.line_size()).fill(E::from_int(0));

    let index_base = b * grad.stride(0) + (c * grad.stride(3));

    for oh in oh_start..oh_end {
        let ih_start = start_index(oh, grad_h, out_h);
        let ih_end = end_index(oh, grad_h, out_h);

        if ih >= ih_start && ih < ih_end {
            for ow in ow_start..ow_end {
                let iw_start = start_index(ow, grad_w, out_w);
                let iw_end = end_index(ow, grad_w, out_w);

                if iw >= iw_start && iw < iw_end {
                    let num_ih = ih_end - ih_start;
                    let num_iw = iw_end - iw_start;

                    let index = index_base + (oh * grad_stride_h) + (ow * grad_stride_w);
                    grad_acc += grad[index / grad.line_size()] / Line::cast_from(num_iw * num_ih);
                }
            }
        }
    }

    output[(b, ih, iw, c)] = grad_acc;
}

#[cube]
fn start_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    (output_size_index * input_size) / output_size
}

#[cube]
fn end_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    let index = (output_size_index + 1) * input_size;
    let index = index.div_ceil(output_size);

    if input_size < index {
        input_size
    } else {
        index
    }
}

pub(crate) fn adaptive_avg_pool2d_backward<R: CubeRuntime>(
    x: CubeTensor<R>,
    out_grad: CubeTensor<R>,
) -> CubeTensor<R> {
    let [batches, channels, height, width] = x.shape.dims();

    let out_grad = into_contiguous_aligned(permute_nchw_to_nhwc(out_grad));
    let line_size = max_line_size(&out_grad);

    let out_shape = Shape::new([batches, height, width, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), out_shape, x.dtype);

    let num_elems = output.shape.num_elements();

    let working_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::new(&x.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&x.client, working_units, cube_dim);

    adaptive_avg_pool2d_backward_direct::launch(
        &x.client,
        cube_count,
        cube_dim,
        out_grad.as_tensor_arg(line_size),
        view4d(&output, line_size),
        shape_divmod(&output),
        ScalarArg::new(working_units),
        output.dtype.into(),
    )
    .expect("Kernel to never fail");

    permute_nhwc_to_nchw(output)
}
