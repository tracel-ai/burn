use crate::{
    CubeRuntime, FloatElement,
    kernel::into_contiguous,
    ops::{empty_qtensor, max_line_size},
};
use crate::{kernel::utils::strided_layout, tensor::CubeTensor};
use burn_tensor::quantization::{QuantInputType, QuantLevel, QuantMode, QuantScheme};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::std::tensor::{StridedLayout, index_offset_contiguous};

#[cube]
fn pack_i8s_to_u32s(value: Line<u32>) -> u32 {
    // NOTE: assuming line size of 4
    let line_size = value.size();
    let mut v_packed = 0;

    #[unroll]
    for i in 0..line_size {
        // Shift and combine into u32
        v_packed |= (value[i] & 0xFF) << (8 * i);
    }
    v_packed
}

#[cube]
fn quantize_symmetric_int8<F: Float>(
    value: Line<F>,
    scale: f32,
    range_min: F,
    range_max: F,
) -> Line<i8> {
    // x_q = clamp(round(x / scale), a, b)
    Line::cast_from(Line::clamp(
        Line::round(value / Line::cast_from(scale)),
        Line::new(range_min),
        Line::new(range_max),
    ))
}

#[cube]
fn quantize_symmetric_int8_packed<F: Float>(
    input: Line<F>,
    scale: f32,
    range_min: F,
    range_max: F,
) -> u32 {
    // Assuming a line size of 4 (equal to the number of values packed)
    // x_q = clamp(round(x / scale), a, b)
    // NOTE: we add 256 before casting to unsigned to correctly represent negative values
    let value = Line::cast_from(
        Line::clamp(
            Line::round(input / Line::cast_from(scale)),
            Line::new(range_min),
            Line::new(range_max),
        ) + Line::cast_from(comptime!(256f32)),
    );
    // Shift and combine into u32
    pack_i8s_to_u32s(value)
}

#[cube]
fn write_scale_per_tensor(in_pos: u32, scale: &Array<f32>, out_scale: &mut Array<f32>) -> f32 {
    let scale = scale[0];

    // Write the scale into the output buffer
    if in_pos == 0 {
        out_scale[in_pos] = scale;
    }

    scale
}

#[cube]
fn write_scale_per_block(
    in_pos: u32,
    scale: &Array<f32>,
    out_scale: &mut Array<f32>,
    #[comptime] block_size: u32,
) -> f32 {
    let scale_pos = in_pos / block_size;
    let scale = scale[scale_pos];

    // Write the scale into the output buffer
    if in_pos % block_size == 0 {
        out_scale[scale_pos] = scale;
    }

    scale
}

#[cube(launch_unchecked)]
fn quantize_per_tensor_symmetric_int8_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    scale: &Array<f32>,
    range_min: F,
    range_max: F,
    output: &mut Tensor<Line<i8>>,
    out_scale: &mut Array<f32>,
    out_layout: StridedLayout,
    #[comptime] rank: Option<u32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let scale = write_scale_per_tensor(ABSOLUTE_POS, scale, out_scale);

    let in_pos = index_offset_contiguous(input, ABSOLUTE_POS, rank);
    let out_pos = out_layout.index(output, ABSOLUTE_POS);

    output[out_pos] = quantize_symmetric_int8(input[in_pos], scale, range_min, range_max);
}

#[cube(launch_unchecked)]
fn quantize_per_block_symmetric_int8_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    scale: &Array<f32>,
    range_min: F,
    range_max: F,
    output: &mut Tensor<Line<i8>>,
    out_scale: &mut Array<f32>,
    out_layout: StridedLayout,
    #[comptime] rank: Option<u32>,
    #[comptime] block_size: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let in_pos = index_offset_contiguous(input, ABSOLUTE_POS, rank);

    let scale = write_scale_per_block(in_pos * input.line_size(), scale, out_scale, block_size);

    let out_pos = out_layout.index(output, ABSOLUTE_POS);
    output[out_pos] = quantize_symmetric_int8(input[in_pos], scale, range_min, range_max);
}

#[cube(launch_unchecked)]
fn quantize_per_tensor_symmetric_int8_packed_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    scale: &Array<f32>,
    range_min: F,
    range_max: F,
    output: &mut Array<u32>,
    out_scale: &mut Array<f32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let scale = write_scale_per_tensor(ABSOLUTE_POS, scale, out_scale);

    if comptime!(input.line_size() == 4) {
        output[ABSOLUTE_POS] =
            quantize_symmetric_int8_packed::<F>(input[ABSOLUTE_POS], scale, range_min, range_max);
    } else {
        // line size 1
        let num_packed = comptime!(4);
        let mut values = Line::<F>::empty(num_packed);
        #[unroll]
        for i in 0..num_packed {
            values[i] = input[ABSOLUTE_POS * num_packed + i][0];
        }
        output[ABSOLUTE_POS] =
            quantize_symmetric_int8_packed::<F>(values, scale, range_min, range_max);
    }
}

#[cube(launch_unchecked)]
fn quantize_per_block_symmetric_int8_packed_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    scale: &Array<f32>,
    range_min: F,
    range_max: F,
    output: &mut Array<u32>,
    out_scale: &mut Array<f32>,
    #[comptime] block_size: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    // line size 1
    let num_packed = comptime!(4);
    let packed_pos = ABSOLUTE_POS * num_packed;
    let scale = write_scale_per_block(packed_pos, scale, out_scale, block_size);

    let mut values = Line::<F>::empty(num_packed);
    #[unroll]
    for i in 0..num_packed {
        values[i] = input[packed_pos + i][0];
    }
    output[ABSOLUTE_POS] = quantize_symmetric_int8_packed::<F>(values, scale, range_min, range_max);
}

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R, F>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
{
    let output = empty_qtensor(tensor.shape.clone(), *scheme, &tensor.device);

    if i8::is_supported(&tensor.client) {
        quantize_unpacked::<R, F>(tensor, scheme, scale, output)
    } else {
        quantize_packed::<R, F>(tensor, scheme, scale, output)
    }
}

fn quantize_unpacked<R: CubeRuntime, F: FloatElement>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
    output: CubeTensor<R>,
) -> CubeTensor<R> {
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let out_layout = strided_layout(&output);
    let out_scale = output.scales().unwrap();

    let line_size = max_line_size(&tensor);
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            ..
        } => {
            unsafe {
                quantize_per_tensor_symmetric_int8_kernel::launch_unchecked::<F, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    tensor.as_tensor_arg::<F>(line_size),
                    scale.as_array_arg::<f32>(1),
                    ScalarArg::new(F::from_int(-i8::MAX as i64)),
                    ScalarArg::new(F::from_int(i8::MAX as i64)),
                    output.as_tensor_arg::<i8>(line_size),
                    out_scale.as_array_arg::<f32>(1),
                    out_layout,
                    Some(tensor.shape.num_dims() as u32),
                )
            };
        }
        QuantScheme {
            level: QuantLevel::Block(block_size),
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            ..
        } => {
            // We could use line_size = block_size if it's in the supported line sizes.. but let's keep it simple
            assert!(
                *block_size as u8 % line_size == 0,
                "block size must be divisible by line size, got block_size={block_size}, line_size={line_size}"
            );
            unsafe {
                quantize_per_block_symmetric_int8_kernel::launch_unchecked::<F, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    tensor.as_tensor_arg::<F>(line_size),
                    scale.as_array_arg::<f32>(1),
                    ScalarArg::new(F::from_int(-i8::MAX as i64)),
                    ScalarArg::new(F::from_int(i8::MAX as i64)),
                    output.as_tensor_arg::<i8>(line_size),
                    out_scale.as_array_arg::<f32>(1),
                    out_layout,
                    Some(tensor.shape.num_dims() as u32),
                    *block_size as u32,
                )
            };
        }
    }

    output
}

fn quantize_packed<R: CubeRuntime, F: FloatElement>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
    output: CubeTensor<R>,
) -> CubeTensor<R> {
    let tensor = into_contiguous(tensor);
    let client = tensor.client.clone();
    // Output tensor contains 4x less elements (four int8 values packed in a single u32)
    let num_elems = tensor.shape.num_elements();

    let out_scale = output.scales().unwrap();

    // Force vectorization to process 4 quantized values packed for 1 output value
    let line_size: u8 = 1;
    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems.div_ceil(line_size as usize), cube_dim);

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            ..
        } => {
            unsafe {
                quantize_per_tensor_symmetric_int8_packed_kernel::launch_unchecked::<F, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    tensor.as_tensor_arg::<F>(line_size),
                    scale.as_array_arg::<f32>(1),
                    ScalarArg::new(F::from_int(-i8::MAX as i64)),
                    ScalarArg::new(F::from_int(i8::MAX as i64)),
                    output.as_array_arg::<u32>(1),
                    out_scale.as_array_arg::<f32>(1),
                )
            };
        }
        QuantScheme {
            level: QuantLevel::Block(block_size),
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            ..
        } => {
            assert!(
                *block_size % 4 == 0,
                "block size must be divisible by 4, got block_size={block_size}"
            );
            unsafe {
                quantize_per_block_symmetric_int8_packed_kernel::launch_unchecked::<F, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    tensor.as_tensor_arg::<F>(line_size),
                    scale.as_array_arg::<f32>(1),
                    ScalarArg::new(F::from_int(-i8::MAX as i64)),
                    ScalarArg::new(F::from_int(i8::MAX as i64)),
                    output.as_array_arg::<u32>(1),
                    out_scale.as_array_arg::<f32>(1),
                    *block_size as u32,
                )
            };
        }
    }

    output
}
