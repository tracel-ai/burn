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

#[cube(launch)]
fn adaptive_avg_pool3d_backward_direct<E: Numeric>(
    grad: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    // Output shape is [batch, depth, height, width, channels] in NDHWC format
    let (out_d, out_h, out_w, channels) = (
        output.shape(1),
        output.shape(2),
        output.shape(3),
        output.shape(4),
    );
    let channel_lines = channels / grad.line_size();
    let (grad_stride_b, grad_stride_d, grad_stride_h, grad_stride_w, grad_stride_c) = (
        grad.stride(0),
        grad.stride(1),
        grad.stride(2),
        grad.stride(3),
        grad.stride(4),
    );
    let (grad_d, grad_h, grad_w) = (grad.shape(1), grad.shape(2), grad.shape(3));

    // Decode position: c, iw, ih, id, b
    let c = (ABSOLUTE_POS % channel_lines) * grad.line_size();
    let pos = ABSOLUTE_POS / channel_lines;
    let iw = pos % out_w;
    let pos = pos / out_w;
    let ih = pos % out_h;
    let pos = pos / out_h;
    let id = pos % out_d;
    let b = pos / out_d;

    let od_start = start_index(id, out_d, grad_d);
    let od_end = end_index(id, out_d, grad_d);

    let oh_start = start_index(ih, out_h, grad_h);
    let oh_end = end_index(ih, out_h, grad_h);

    let ow_start = start_index(iw, out_w, grad_w);
    let ow_end = end_index(iw, out_w, grad_w);

    let mut grad_acc = Line::empty(grad.line_size()).fill(E::from_int(0));

    let index_base = b * grad_stride_b + (c * grad_stride_c);

    for od in od_start..od_end {
        let id_start = start_index(od, grad_d, out_d);
        let id_end = end_index(od, grad_d, out_d);

        if id >= id_start && id < id_end {
            for oh in oh_start..oh_end {
                let ih_start = start_index(oh, grad_h, out_h);
                let ih_end = end_index(oh, grad_h, out_h);

                if ih >= ih_start && ih < ih_end {
                    for ow in ow_start..ow_end {
                        let iw_start = start_index(ow, grad_w, out_w);
                        let iw_end = end_index(ow, grad_w, out_w);

                        if iw >= iw_start && iw < iw_end {
                            let num_id = id_end - id_start;
                            let num_ih = ih_end - ih_start;
                            let num_iw = iw_end - iw_start;

                            let index = index_base
                                + (od * grad_stride_d)
                                + (oh * grad_stride_h)
                                + (ow * grad_stride_w);
                            grad_acc += grad[index / grad.line_size()]
                                / Line::cast_from(num_id * num_ih * num_iw);
                        }
                    }
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

pub(crate) fn adaptive_avg_pool3d_backward<R: CubeRuntime>(
    x: CubeTensor<R>,
    out_grad: CubeTensor<R>,
) -> CubeTensor<R> {
    let [batches, channels, depth, height, width] = x.shape.dims();

    let out_grad = into_contiguous(permute_ncdhw_to_ndhwc(out_grad));
    let line_size = max_line_size(&out_grad);

    let out_shape = Shape::new([batches, depth, height, width, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), out_shape, x.dtype);

    let num_elems = output.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    adaptive_avg_pool3d_backward_direct::launch(
        &x.client,
        cube_count,
        cube_dim,
        out_grad.as_tensor_arg(line_size),
        output.as_tensor_arg(line_size),
        output.dtype.into(),
    )
    .expect("Kernel to never fail");

    permute_ndhwc_to_ncdhw(output)
}
