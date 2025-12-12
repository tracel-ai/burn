use crate::{
    CubeRuntime,
    kernel::into_contiguous,
    ops::{max_line_size, numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_backend::Shape;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch)]
fn adaptive_avg_pool2d_direct<E: Numeric>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let (out_h, out_w, channels) = (output.shape(1), output.shape(2), output.shape(3));
    let channel_lines = channels / output.line_size();
    let (in_stride_b, in_stride_h, in_stride_w, in_stride_c) = (
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
    );
    let (in_h, in_w) = (input.shape(1), input.shape(2));

    let c = (ABSOLUTE_POS % channel_lines) * input.line_size();
    let pos = ABSOLUTE_POS / channel_lines;
    let ow = pos % out_w;
    let pos = pos / out_w;
    let oh = pos % out_h;
    let b = pos / out_h;

    let ih_start = start_index(oh, out_h, in_h);
    let ih_end = end_index(oh, out_h, in_h);

    let iw_start = start_index(ow, out_w, in_w);
    let iw_end = end_index(ow, out_w, in_w);

    let mut sum = Line::empty(input.line_size()).fill(E::from_int(0));

    let index_input_0 = b * in_stride_b;
    let index_input_1 = c * in_stride_c;

    for ih in ih_start..ih_end {
        let index_input_2 = ih * in_stride_h;

        for iw in iw_start..iw_end {
            let index_input_3 = iw * in_stride_w;

            let index_input = index_input_0 + index_input_1 + index_input_2 + index_input_3;
            sum += input[index_input / input.line_size()];
        }
    }

    let num_ih = ih_end - ih_start;
    let num_iw = iw_end - iw_start;

    output[ABSOLUTE_POS] = sum / Line::cast_from(num_ih * num_iw);
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

pub(crate) fn adaptive_avg_pool2d<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
) -> CubeTensor<R> {
    let [batch_size, channels, _, _] = input.shape.dims();

    let input = into_contiguous(permute_nchw_to_nhwc(input));
    let line_size = max_line_size(&input);

    let output_shape = Shape::new([batch_size, output_size[0], output_size[1], channels]);
    let num_elems: usize = output_shape.num_elements();
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    adaptive_avg_pool2d_direct::launch(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(line_size),
        output.as_tensor_arg(line_size),
        output.dtype.into(),
    )
    .expect("Kernel to never fail");

    permute_nhwc_to_nchw(output)
}
