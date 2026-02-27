use crate::{
    CubeRuntime,
    kernel::{
        into_contiguous_aligned,
        pool::pool2d::{Position, view4d},
        utils::{address_type, decompose_linear, shape_divmod},
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

#[cube(launch, address_type = "dynamic")]
fn adaptive_avg_pool2d_direct<E: Numeric>(
    input: &Tensor<Line<E>>,
    output: &mut View<Line<E>, Position, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (_, pos) = decompose_linear(ABSOLUTE_POS * output.line_size(), &out_shape);
    let [b, oh, ow, c] = *pos else { unreachable!() };

    let (_, out_h, out_w, _) = output.shape();
    let (in_stride_h, in_stride_w) = (input.stride(1), input.stride(2));
    let (in_h, in_w) = (input.shape(1), input.shape(2));

    let ih_start = start_index(oh, out_h, in_h);
    let ih_end = end_index(oh, out_h, in_h);

    let iw_start = start_index(ow, out_w, in_w);
    let iw_end = end_index(ow, out_w, in_w);

    let mut sum = Line::empty(input.line_size()).fill(E::from_int(0));

    let index_input_base = b * input.stride(0) + c * input.stride(3);

    for ih in ih_start..ih_end {
        let index_input_2 = ih * in_stride_h;

        for iw in iw_start..iw_end {
            let index_input_3 = iw * in_stride_w;

            let index_input = index_input_base + index_input_2 + index_input_3;
            sum += input[index_input / input.line_size()];
        }
    }

    let num_ih = ih_end - ih_start;
    let num_iw = iw_end - iw_start;

    output[(b, oh, ow, c)] = sum / Line::cast_from(num_ih * num_iw);
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

pub(crate) fn adaptive_avg_pool2d<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
) -> CubeTensor<R> {
    let [batch_size, channels, _, _] = input.meta.shape().dims();

    let input = into_contiguous_aligned(permute_nchw_to_nhwc(input));
    let line_size = max_line_size(&input);

    let output_shape = Shape::new([batch_size, output_size[0], output_size[1], channels]);
    let num_elems: usize = output_shape.num_elements();
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );

    let working_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    adaptive_avg_pool2d_direct::launch(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(input, output),
        input.into_tensor_arg(line_size),
        view4d(output.clone(), line_size),
        shape_divmod(&output),
        ScalarArg::new(working_units),
        output.dtype.into(),
    );

    permute_nhwc_to_nchw(output)
}
