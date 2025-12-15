use crate::{
    CubeRuntime,
    kernel::into_contiguous,
    ops::{max_line_size, numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_backend::Shape;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch)]
fn adaptive_avg_pool2d_backward_direct<E: Numeric>(
    grad: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let (out_h, out_w, channels) = (output.shape(1), output.shape(2), output.shape(3));
    let channel_lines = channels / grad.line_size();
    let (grad_stride_b, grad_stride_h, grad_stride_w, grad_stride_c) = (
        grad.stride(0),
        grad.stride(1),
        grad.stride(2),
        grad.stride(3),
    );
    let (grad_h, grad_w) = (grad.shape(1), grad.shape(2));

    let c = (ABSOLUTE_POS % channel_lines) * grad.line_size();
    let pos = ABSOLUTE_POS / channel_lines;
    let iw = pos % out_w;
    let pos = pos / out_w;
    let ih = pos % out_h;
    let b = pos / out_h;

    let oh_start = start_index(ih, out_h, grad_h);
    let oh_end = end_index(ih, out_h, grad_h);

    let ow_start = start_index(iw, out_w, grad_w);
    let ow_end = end_index(iw, out_w, grad_w);

    let mut grad_acc = Line::empty(grad.line_size()).fill(E::from_int(0));

    let index_base = b * grad_stride_b + (c * grad_stride_c);

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

    output[ABSOLUTE_POS] = grad_acc;
}

#[cube]
fn start_index(output_size_index: u32, output_size: u32, input_size: u32) -> u32 {
    (output_size_index * input_size) / output_size
}

#[cube]
fn end_index(output_size_index: u32, output_size: u32, input_size: u32) -> u32 {
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

    let out_grad = into_contiguous(permute_nchw_to_nhwc(out_grad));
    let line_size = max_line_size(&out_grad);

    let out_shape = Shape::new([batches, height, width, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), out_shape, x.dtype);

    let num_elems = output.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    adaptive_avg_pool2d_backward_direct::launch(
        &x.client,
        cube_count,
        cube_dim,
        out_grad.as_tensor_arg(line_size),
        output.as_tensor_arg(line_size),
        output.dtype.into(),
    )
    .expect("Kernel to never fail");

    permute_nhwc_to_nchw(output)
}
