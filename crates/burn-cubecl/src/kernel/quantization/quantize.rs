use crate::tensor::CubeTensor;
use crate::{CubeElement, CubeRuntime, IntElement};
use burn_tensor::Shape;
use burn_tensor::quantization::{
    BlockLayout, QuantizationMode, QuantizationScheme, QuantizationType,
};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

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
fn quantize_affine_int8<F: Float>(
    value: Line<F>,
    scale: f32,
    offset: i32,
    range_min: f32,
    range_max: f32,
) -> Line<u32> {
    // x_q = clamp(round(x / scale + offset), a, b)
    // NOTE: we add 256 before casting to unsigned to correctly represent negative values
    Line::cast_from(
        Line::clamp(
            Line::round((value / Line::cast_from(scale)) + Line::cast_from(offset)),
            Line::cast_from(range_min),
            Line::cast_from(range_max),
        ) + Line::cast_from(comptime!(256f32)),
    )
}

#[cube]
fn quantize_symmetric_int8<F: Float>(
    value: Line<F>,
    scale: f32,
    range_min: F,
    range_max: F,
) -> Line<u32> {
    // x_q = clamp(round(x / scale), a, b)
    // NOTE: we add 256 before casting to unsigned to correctly represent negative values
    Line::cast_from(
        Line::clamp(
            Line::round(value / Line::cast_from(scale)),
            Line::new(range_min),
            Line::new(range_max),
        ) + Line::cast_from(comptime!(256f32)),
    )
}

#[cube]
fn quantize_affine_int8_packed(
    input: Line<f32>,
    scale: f32,
    offset: i32,
    range_min: f32,
    range_max: f32,
) -> u32 {
    // Assuming a line size of 4 (equal to the number of values packed)
    let value = quantize_affine_int8::<f32>(input, scale, offset, range_min, range_max);
    // Shift and combine into u32
    pack_i8s_to_u32s(value)
}

#[cube]
fn quantize_symmetric_int8_packed(
    input: Line<f32>,
    scale: f32,
    range_min: f32,
    range_max: f32,
) -> u32 {
    // Assuming a line size of 4 (equal to the number of values packed)
    let value = quantize_symmetric_int8::<f32>(input, scale, range_min, range_max);
    // Shift and combine into u32
    pack_i8s_to_u32s(value)
}

#[cube(launch_unchecked)]
fn quantize_per_tensor_affine_int8_kernel(
    input: &Tensor<Line<f32>>,
    scale: &Tensor<f32>,
    offset: &Tensor<i32>,
    range_min: f32,
    range_max: f32,
    output: &mut Array<u32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let scale = scale[0];
    let offset = offset[0];

    // Cast the scale to u32 and write the value in the output
    if ABSOLUTE_POS == output.len() - 1 {
        output[ABSOLUTE_POS] = u32::reinterpret(scale);
        terminate!();
    }

    // Cast the offset to u32 and write the value in the output
    if ABSOLUTE_POS == output.len() - 2 {
        output[ABSOLUTE_POS] = u32::reinterpret(offset);
        terminate!();
    }

    if comptime!(input.line_size() == 4) {
        output[ABSOLUTE_POS] =
            quantize_affine_int8_packed(input[ABSOLUTE_POS], scale, offset, range_min, range_max);
    } else {
        // line size 1
        let num_packed = comptime!(4);
        let mut values = Line::<f32>::empty(num_packed);
        #[unroll]
        for i in 0..num_packed {
            values[i] = input[ABSOLUTE_POS + i][0];
        }
        output[ABSOLUTE_POS] =
            quantize_affine_int8_packed(values, scale, offset, range_min, range_max);
    }
}

// Would have wrapped symmetric with the same affine kernel but cube doesn't support Option<Tensor> for offset.
#[cube(launch_unchecked)]
fn quantize_per_tensor_symmetric_int8_kernel(
    input: &Tensor<Line<f32>>,
    scale: &Tensor<f32>,
    range_min: f32,
    range_max: f32,
    output: &mut Array<u32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let scale = scale[0];

    // Cast the scale to u32 and write the value in the output
    if ABSOLUTE_POS == output.len() - 1 {
        output[ABSOLUTE_POS] = u32::reinterpret(scale);
        terminate!();
    }

    if comptime!(input.line_size() == 4) {
        output[ABSOLUTE_POS] =
            quantize_symmetric_int8_packed(input[ABSOLUTE_POS], scale, range_min, range_max);
    } else {
        // line size 1
        let num_packed = comptime!(4);
        let mut values = Line::<f32>::empty(num_packed);
        #[unroll]
        for i in 0..num_packed {
            values[i] = input[ABSOLUTE_POS + i][0];
        }
        output[ABSOLUTE_POS] = quantize_symmetric_int8_packed(values, scale, range_min, range_max);
    }
}

#[cube(launch_unchecked)]
fn quantize_per_block_flat_symmetric_int8_kernel(
    input: &Tensor<Line<f32>>,
    scale: &Tensor<f32>,
    range_min: f32,
    range_max: f32,
    block_size: u32,
    output: &mut Array<u32>,
    #[comptime] num_blocks: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    // Cast the scale to u32 and write the value in the output
    if ABSOLUTE_POS >= output.len() - num_blocks {
        let scale_idx = num_blocks - (output.len() - ABSOLUTE_POS);
        output[ABSOLUTE_POS] = u32::reinterpret(scale[scale_idx]);
        terminate!();
    }

    let line_size = comptime!(input.line_size());
    let block_idx = (ABSOLUTE_POS * line_size) / block_size;
    let scale = scale[block_idx];
    if comptime!(line_size == 4) {
        output[ABSOLUTE_POS] =
            quantize_symmetric_int8_packed(input[ABSOLUTE_POS], scale, range_min, range_max);
    }
}

#[cube(launch_unchecked)]
fn quantize_per_block_flat_affine_int8_kernel(
    input: &Tensor<Line<f32>>,
    scale: &Tensor<f32>,
    offset: &Tensor<i32>,
    range_min: f32,
    range_max: f32,
    block_size: u32,
    output: &mut Array<u32>,
    #[comptime] num_blocks: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    // Cast the scale to u32 and write the value in the output
    if ABSOLUTE_POS >= output.len() - num_blocks {
        let scale_idx = num_blocks - (output.len() - ABSOLUTE_POS);
        output[ABSOLUTE_POS] = u32::reinterpret(scale[scale_idx]);
        terminate!();
    }

    // Cast the offset to u32 and write the value in the output
    if ABSOLUTE_POS >= output.len() - 2 * num_blocks {
        let offset_idx = 2 * num_blocks - (output.len() - ABSOLUTE_POS);
        output[ABSOLUTE_POS] = u32::reinterpret(offset[offset_idx]);
        terminate!();
    }

    let line_size = comptime!(input.line_size());
    let block_idx = (ABSOLUTE_POS * line_size) / block_size;
    let scale = scale[block_idx];
    let offset = offset[block_idx];
    if comptime!(line_size == 4) {
        output[ABSOLUTE_POS] =
            quantize_affine_int8_packed(input[ABSOLUTE_POS], scale, offset, range_min, range_max);
    }
}

fn create_quantized_output<R: CubeRuntime>(
    client: ComputeClient<R::Server, R::Channel>,
    num_input_elems: usize,
    device: R::Device,
    shape: Shape,
    scheme: QuantizationScheme,
) -> CubeTensor<R> {
    // Output tensor contains 4x less elements (four int8 values packed in a single u32)
    let output_elems_size = usize::div_ceil(num_input_elems, 4) * core::mem::size_of::<u32>();

    // Scale and offset (optional) qparams are also packed in the tensor data
    let qparams_size = match &scheme {
        QuantizationScheme::PerTensor(mode, ..) => match mode {
            QuantizationMode::Affine => core::mem::size_of::<f32>() + core::mem::size_of::<i32>(),
            QuantizationMode::Symmetric => core::mem::size_of::<f32>(),
        },
        QuantizationScheme::PerBlock(mode, _, layout) => {
            let num_blocks = match layout {
                BlockLayout::Flat(block_size) => num_input_elems / *block_size as usize,
                BlockLayout::Grid(m, n) => num_input_elems / (m * n) as usize,
            };

            match mode {
                QuantizationMode::Affine => {
                    (core::mem::size_of::<f32>() + core::mem::size_of::<i32>()) * num_blocks
                }
                QuantizationMode::Symmetric => core::mem::size_of::<f32>() * num_blocks,
            }
        }
    };

    let handle = client.empty(output_elems_size + qparams_size);
    CubeTensor::new_contiguous(
        client,
        device,
        shape,
        handle,
        burn_tensor::DType::QFloat(scheme),
    )
}

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R, F, I>(
    tensor: CubeTensor<R>,
    scheme: &QuantizationScheme,
    scale: CubeTensor<R>,
    offset: Option<CubeTensor<R>>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: CubeElement,
    I: IntElement,
{
    let client = tensor.client.clone();
    // Output tensor contains 4x less elements (four int8 values packed in a single u32)
    let num_elems = tensor.shape.num_elements();

    // Force vectorization to process 4 quantized values packed for 1 output value
    let line_size: u8 = if num_elems < 4 { 1 } else { 4 };
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    let output = create_quantized_output(
        client.clone(),
        num_elems,
        tensor.device.clone(),
        tensor.shape.clone(),
        *scheme,
    );

    match scheme {
        QuantizationScheme::PerTensor(mode, QuantizationType::QInt8) => {
            let ndims = tensor.shape.num_dims();
            let dummy_array = vec![1; ndims];

            match mode {
                QuantizationMode::Affine => {
                    unsafe {
                        quantize_per_tensor_affine_int8_kernel::launch_unchecked::<R>(
                            &client,
                            cube_count,
                            cube_dim,
                            tensor.as_tensor_arg::<F>(line_size),
                            // Ignore shape and stride
                            TensorArg::from_raw_parts::<F>(
                                &scale.handle,
                                &dummy_array,
                                &dummy_array,
                                1,
                            ),
                            TensorArg::from_raw_parts::<I>(
                                &offset.expect("Should have offset").handle,
                                &dummy_array,
                                &dummy_array,
                                1,
                            ),
                            ScalarArg::new(i8::MIN as f32),
                            ScalarArg::new(i8::MAX as f32),
                            output.as_array_arg::<u32>(1),
                        )
                    };
                }
                QuantizationMode::Symmetric => {
                    unsafe {
                        quantize_per_tensor_symmetric_int8_kernel::launch_unchecked::<R>(
                            &client,
                            cube_count,
                            cube_dim,
                            tensor.as_tensor_arg::<F>(line_size),
                            // Ignore shape and stride
                            TensorArg::from_raw_parts::<F>(
                                &scale.handle,
                                &dummy_array,
                                &dummy_array,
                                1,
                            ),
                            ScalarArg::new(-i8::MAX as f32),
                            ScalarArg::new(i8::MAX as f32),
                            output.as_array_arg::<u32>(1),
                        )
                    };
                }
            }
        }
        QuantizationScheme::PerBlock(
            mode,
            QuantizationType::QInt8,
            BlockLayout::Flat(block_size),
        ) => {
            if line_size != 4 {
                panic!(
                    "Per-block quantization is only supported for a line size of 4, got {line_size} ({num_elems} elements)"
                )
            }

            if block_size % line_size as u32 != 0 {
                panic!("Block size must be a factor of {line_size}, got {block_size}")
            }

            let num_blocks = num_elems as u32 / block_size;
            match mode {
                QuantizationMode::Affine => {
                    unsafe {
                        quantize_per_block_flat_affine_int8_kernel::launch_unchecked::<R>(
                            &client,
                            cube_count,
                            cube_dim,
                            tensor.as_tensor_arg::<F>(line_size),
                            scale.as_tensor_arg::<F>(1),
                            offset.expect("Should have offset").as_tensor_arg::<I>(1),
                            ScalarArg::new(i8::MIN as f32),
                            ScalarArg::new(i8::MAX as f32),
                            ScalarArg::new(*block_size),
                            output.as_array_arg::<u32>(1),
                            num_blocks,
                        )
                    };
                }
                QuantizationMode::Symmetric => {
                    unsafe {
                        quantize_per_block_flat_symmetric_int8_kernel::launch_unchecked::<R>(
                            &client,
                            cube_count,
                            cube_dim,
                            tensor.as_tensor_arg::<F>(line_size),
                            scale.as_tensor_arg::<F>(1),
                            ScalarArg::new(-i8::MAX as f32),
                            ScalarArg::new(i8::MAX as f32),
                            ScalarArg::new(*block_size),
                            output.as_array_arg::<u32>(1),
                            num_blocks,
                        )
                    };
                }
            }
        }
        QuantizationScheme::PerBlock(.., BlockLayout::Grid(..)) => {
            panic!("Per-block quantization is not supported for grid layout")
        }
    }

    output
}
