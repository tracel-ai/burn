use crate::{element::CubeElement, tensor::CubeTensor, CubeRuntime};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch)]
fn adaptive_avg_pool2d_backward_direct<E: Numeric>(grad: &Tensor<E>, output: &mut Tensor<E>) {
    let (output_stride_0, output_stride_1, output_stride_2, output_stride_3) = (
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
    );
    let (output_shape_0, output_shape_1, output_shape_2, output_shape_3) = (
        output.shape(0),
        output.shape(1),
        output.shape(2),
        output.shape(3),
    );
    let (grad_stride_0, grad_stride_1, grad_stride_2, grad_stride_3) = (
        grad.stride(0),
        grad.stride(1),
        grad.stride(2),
        grad.stride(3),
    );
    let (grad_shape_2, grad_shape_3) = (grad.shape(2), grad.shape(3));

    let b = (ABSOLUTE_POS / output_stride_0) % output_shape_0;
    let c = (ABSOLUTE_POS / output_stride_1) % output_shape_1;
    let ih = (ABSOLUTE_POS / output_stride_2) % output_shape_2;
    let iw = (ABSOLUTE_POS / output_stride_3) % output_shape_3;

    let oh_start = start_index(ih, output_shape_2, grad_shape_2);
    let oh_end = end_index(ih, output_shape_2, grad_shape_2);

    let ow_start = start_index(iw, output_shape_3, grad_shape_3);
    let ow_end = end_index(iw, output_shape_3, grad_shape_3);

    let mut grad_acc = E::from_int(0);

    let index_base = b * grad_stride_0 + (c * grad_stride_1);

    for oh in oh_start..oh_end {
        let ih_start = start_index(oh, grad_shape_2, output_shape_2);
        let ih_end = end_index(oh, grad_shape_2, output_shape_2);

        if ih >= ih_start && ih < ih_end {
            for ow in ow_start..ow_end {
                let iw_start = start_index(ow, grad_shape_3, output_shape_3);
                let iw_end = end_index(ow, grad_shape_3, output_shape_3);

                if iw >= iw_start && iw < iw_end {
                    let num_ih = ih_end - ih_start;
                    let num_iw = iw_end - iw_start;

                    let index = index_base + (oh * grad_stride_2) + (ow * grad_stride_3);
                    grad_acc += grad[index] / E::cast_from(num_iw * num_ih);
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

#[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
#[allow(clippy::manual_div_ceil)]
#[cube]
fn end_index(output_size_index: u32, output_size: u32, input_size: u32) -> u32 {
    let index = (output_size_index + 1) * input_size;
    let index = (index + output_size - 1) / output_size;

    if input_size < index {
        input_size
    } else {
        index
    }
}

pub(crate) fn adaptive_avg_pool2d_backward<R: CubeRuntime, E: CubeElement>(
    x: CubeTensor<R>,
    out_grad: CubeTensor<R>,
) -> CubeTensor<R> {
    let output_shape = x.shape.clone();
    let num_elems = output_shape.num_elements();
    let output_buffer = x.client.empty(num_elems * core::mem::size_of::<E>());
    let output = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        output_shape,
        output_buffer,
        x.dtype,
    );

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    adaptive_avg_pool2d_backward_direct::launch::<E, R>(
        &x.client,
        cube_count,
        cube_dim,
        out_grad.as_tensor_arg::<E>(1),
        output.as_tensor_arg::<E>(1),
    );

    output
}
