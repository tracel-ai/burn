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
fn adaptive_avg_pool3d_direct<E: Numeric>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    // Output shape is [batch, out_d, out_h, out_w, channels] in NDHWC format
    let (out_d, out_h, out_w, channels) = (
        output.shape(1),
        output.shape(2),
        output.shape(3),
        output.shape(4),
    );
    let channel_lines = channels / output.line_size();
    let (in_stride_b, in_stride_d, in_stride_h, in_stride_w, in_stride_c) = (
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        input.stride(4),
    );
    let (in_d, in_h, in_w) = (input.shape(1), input.shape(2), input.shape(3));

    // Decode position: c, ow, oh, od, b
    let c = (ABSOLUTE_POS % channel_lines) * input.line_size();
    let pos = ABSOLUTE_POS / channel_lines;
    let ow = pos % out_w;
    let pos = pos / out_w;
    let oh = pos % out_h;
    let pos = pos / out_h;
    let od = pos % out_d;
    let b = pos / out_d;

    let id_start = start_index(od, out_d, in_d);
    let id_end = end_index(od, out_d, in_d);

    let ih_start = start_index(oh, out_h, in_h);
    let ih_end = end_index(oh, out_h, in_h);

    let iw_start = start_index(ow, out_w, in_w);
    let iw_end = end_index(ow, out_w, in_w);

    let mut sum = Line::empty(input.line_size()).fill(E::from_int(0));

    let index_input_0 = b * in_stride_b;
    let index_input_1 = c * in_stride_c;

    for id in id_start..id_end {
        let index_input_2 = id * in_stride_d;

        for ih in ih_start..ih_end {
            let index_input_3 = ih * in_stride_h;

            for iw in iw_start..iw_end {
                let index_input_4 = iw * in_stride_w;

                let index_input =
                    index_input_0 + index_input_1 + index_input_2 + index_input_3 + index_input_4;
                sum += input[index_input / input.line_size()];
            }
        }
    }

    let num_id = id_end - id_start;
    let num_ih = ih_end - ih_start;
    let num_iw = iw_end - iw_start;

    output[ABSOLUTE_POS] = sum / Line::cast_from(num_id * num_ih * num_iw);
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

pub(crate) fn adaptive_avg_pool3d<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 3],
) -> CubeTensor<R> {
    let [batch_size, channels, _, _, _] = input.shape.dims();

    let input = into_contiguous(permute_ncdhw_to_ndhwc(input));
    let line_size = max_line_size(&input);

    let output_shape = Shape::new([
        batch_size,
        output_size[0],
        output_size[1],
        output_size[2],
        channels,
    ]);
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

    adaptive_avg_pool3d_direct::launch(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(line_size),
        output.as_tensor_arg(line_size),
        output.dtype.into(),
    )
    .expect("Kernel to never fail");

    permute_ndhwc_to_ncdhw(output)
}
