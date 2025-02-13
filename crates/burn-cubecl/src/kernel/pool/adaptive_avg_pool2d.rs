use crate::{element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor, CubeRuntime};
use burn_tensor::Shape;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch)]
fn adaptive_avg_pool2d_direct<E: Numeric>(input: &Tensor<E>, output: &mut Tensor<E>) {
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
    let (input_stride_0, input_stride_1, input_stride_2, input_stride_3) = (
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
    );
    let (input_shape_2, input_shape_3) = (input.shape(2), input.shape(3));

    let b = (ABSOLUTE_POS / output_stride_0) % output_shape_0;
    let c = (ABSOLUTE_POS / output_stride_1) % output_shape_1;
    let oh = (ABSOLUTE_POS / output_stride_2) % output_shape_2;
    let ow = (ABSOLUTE_POS / output_stride_3) % output_shape_3;

    let ih_start = start_index(oh, output_shape_2, input_shape_2);
    let ih_end = end_index(oh, output_shape_2, input_shape_2);

    let iw_start = start_index(ow, output_shape_3, input_shape_3);
    let iw_end = end_index(ow, output_shape_3, input_shape_3);

    let mut sum = E::from_int(0);

    let index_input_0 = b * input_stride_0;
    let index_input_1 = c * input_stride_1;

    for ih in ih_start..ih_end {
        let index_input_2 = ih * input_stride_2;

        for iw in iw_start..iw_end {
            let index_input_3 = iw * input_stride_3;

            let index_input = index_input_0 + index_input_1 + index_input_2 + index_input_3;
            sum += input[index_input];
        }
    }

    let num_ih = ih_end - ih_start;
    let num_iw = iw_end - iw_start;

    output[ABSOLUTE_POS] = sum / E::cast_from(num_ih * num_iw);
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

pub(crate) fn adaptive_avg_pool2d<R: CubeRuntime, E: CubeElement>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
) -> CubeTensor<R> {
    let [batch_size, channels, _, _] = input.shape.dims();

    let output_shape = Shape::new([batch_size, channels, output_size[0], output_size[1]]);
    let num_elems: usize = output_shape.num_elements();
    let output = empty_device::<R, E>(input.client.clone(), input.device.clone(), output_shape);

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    adaptive_avg_pool2d_direct::launch::<E, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<E>(1),
        output.as_tensor_arg::<E>(1),
    );

    output
}
