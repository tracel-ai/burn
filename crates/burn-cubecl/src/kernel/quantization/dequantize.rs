#![allow(missing_docs)] // pub cube modules

use super::QParams;
use crate::{CubeRuntime, FloatElement, kernel::utils::strided_layout, ops::max_line_size};
use crate::{ops::numeric::empty_device_strided, tensor::CubeTensor};
use burn_tensor::DType;
use burn_tensor::quantization::{QuantInputType, QuantLevel, QuantMode, QuantScheme};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::std::tensor::{StridedLayout, index_offset_contiguous};

/// Dequantize a line of values into floating-point values using the provided scale.
#[cube]
pub fn dequantize_symmetric<I: Int, F: Float>(value: Line<I>, scale: f32) -> Line<F> {
    // x = scale * x_q
    Line::cast_from(scale) * Line::cast_from(value)
}

/// Dequantize the value at a specified position using the provided quantization scheme.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_values<F: Float, QI: Int>(
    position: u32,
    values: &Tensor<QI>,
    scales: &Tensor<f32>,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    let value = values[position];
    dequantize_value::<F, QI>(position, value, scales, scheme)
}

/// Dequantize a single value using the scale at the specified position.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_value<F: Float, QI: Int>(
    position: u32,
    value: QI,
    scales: &Tensor<f32>,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    let qparams = QParams::new(scheme);
    let scale = qparams.scale(scales, position);
    // TODO: q_store_type: QuantStoreType::Native
    let floats = unpack_q::<F, QI>(value, scheme.q_type);

    floats * Line::cast_from(scale)
}

/// Unpack a quantized integer into a line of floating-point values, according to the specified quantization input type.
///
/// This handles types where multiple quantized values are packed into a single integer (the stored quantization type).
#[allow(clippy::explicit_counter_loop)]
#[cube]
fn unpack_q<F: Float, QI: Int>(value: QI, #[comptime] quant: QuantInputType) -> Line<F> {
    let size_quant = comptime!(match quant {
        QuantInputType::QInt8 => 8,
    });
    let size_store = comptime!(QI::size_bits().unwrap() as u32);
    let num_quant = comptime!(size_store / size_quant);

    let mut empty = Line::empty(num_quant);
    let mut position = comptime!(0);
    let mask = QI::cast_from(comptime!((1 << size_quant) - 1));

    #[unroll]
    for _ in 0..num_quant {
        let offset = QI::cast_from(comptime!(position * size_store));
        let value = (value >> offset) & mask;
        empty[position] = F::cast_from(value);
        comptime!(position += 1);
    }

    empty
}

#[cube]
fn extract_i8(value: u32, offset: u32) -> i32 {
    // Extract 8-bit segment
    let value = (value >> offset) & 0xFF;
    // Check if the value is negative by inspecting the MSB and subtract 256 if it is
    // Subtract 0 or 256 to circumvent unsupported conditional assignment (let x = if {} else {};)
    let sub = i32::cast_from(value & 0x80 != 0) * 256;
    i32::cast_from(value) - sub
}

#[cube]
fn unpack_i8s(value: u32) -> Line<i32> {
    let mut line = Line::empty(4_u32);
    // Extract each 8-bit segment
    line[0] = extract_i8(value, 0);
    line[1] = extract_i8(value, 8);
    line[2] = extract_i8(value, 16);
    line[3] = extract_i8(value, 24);

    line
}

#[cube(launch_unchecked)]
fn dequantize_per_tensor_symmetric_int8_packed_kernel<F: Float>(
    input: &Tensor<Line<u32>>,
    scale: &Tensor<f32>,
    output: &mut Tensor<Line<F>>,
    #[comptime] scheme: QuantScheme,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    let qparams = QParams::new(scheme);
    let scale = qparams.scale(scale, 0);

    let value = input[ABSOLUTE_POS];

    // Input line size is fixed to 1
    if comptime!(output.line_size() == 4) {
        output[ABSOLUTE_POS] = dequantize_symmetric(unpack_i8s(value[0]), scale);
    } else {
        // For very small inputs where number of elements < 4, the output line size is 1
        let out = dequantize_symmetric::<i32, F>(unpack_i8s(value[0]), scale);

        #[unroll]
        for j in 0..out.size() {
            output[ABSOLUTE_POS * out.size() + j] = Line::cast_from(out[j]);
        }
    }
}

#[cube(launch_unchecked)]
fn dequantize_per_block_symmetric_int8_packed_kernel<F: Float>(
    input: &Tensor<Line<u32>>,
    scale: &Tensor<f32>,
    output: &mut Tensor<Line<F>>,
    #[comptime] scheme: QuantScheme,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    let qparams = QParams::new(scheme);
    let scale = qparams.scale(scale, ABSOLUTE_POS);

    let value = input[ABSOLUTE_POS];

    // Input line size is fixed to 1
    if comptime!(output.line_size() == 4) {
        output[ABSOLUTE_POS] = dequantize_symmetric(unpack_i8s(value[0]), scale);
    } else {
        // For very small inputs where number of elements < 4, the output line size is 1
        let out = dequantize_symmetric::<i32, F>(unpack_i8s(value[0]), scale);

        #[unroll]
        for j in 0..out.size() {
            output[ABSOLUTE_POS * out.size() + j] = Line::cast_from(out[j]);
        }
    }
}

#[cube(launch_unchecked)]
fn dequantize_per_tensor_symmetric_int8_native_kernel<F: Float>(
    input: &Tensor<Line<i8>>,
    scale: &Tensor<f32>,
    output: &mut Tensor<Line<F>>,
    out_layout: StridedLayout,
    #[comptime] scheme: QuantScheme,
    #[comptime] rank: Option<u32>,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    let in_pos = index_offset_contiguous(input, ABSOLUTE_POS, rank);
    let out_pos = out_layout.index(output, ABSOLUTE_POS);

    let qparams = QParams::new(scheme);
    let scale = qparams.scale(scale, 0);

    output[out_pos] = dequantize_symmetric(input[in_pos], scale);
}

#[cube(launch_unchecked)]
fn dequantize_per_block_symmetric_int8_native_kernel<F: Float>(
    input: &Tensor<Line<i8>>,
    scale: &Tensor<f32>,
    output: &mut Tensor<Line<F>>,
    out_layout: StridedLayout,
    #[comptime] scheme: QuantScheme,
    #[comptime] rank: Option<u32>,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    let in_pos = index_offset_contiguous(input, ABSOLUTE_POS, rank);
    let out_pos = out_layout.index(output, ABSOLUTE_POS);

    let qparams = QParams::new(scheme);
    // Absolute pos represents the logical block (scale) used to dequantize, not layout
    let scale = qparams.scale(scale, ABSOLUTE_POS * input.line_size());

    output[out_pos] = dequantize_symmetric(input[in_pos], scale);
}

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R, F>(tensor: CubeTensor<R>) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
{
    let shape = tensor.shape.clone();
    let output = empty_device_strided::<R, F>(tensor.client.clone(), tensor.device.clone(), shape);

    if i8::is_supported(&tensor.client) {
        dequantize_native::<R, F>(tensor, output)
    } else {
        dequantize_packed::<R, F>(tensor, output)
    }
}

fn dequantize_packed<R, F>(tensor: CubeTensor<R>, output: CubeTensor<R>) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
{
    // The actual number of elements is 1/4 (four int8 values packed in a single u32)
    // so we choose a line size to match a valid input binding size.
    let num_out_elems = tensor.shape.num_elements();
    let num_elems = usize::div_ceil(num_out_elems, 4);
    let line_size_in = 1;
    let line_size_out = 1;
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size_in as usize, cube_dim);

    if let DType::QFloat(scheme) = tensor.dtype {
        match scheme {
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                q_type: QuantInputType::QInt8,
                ..
            } => {
                let scales = tensor.scales().unwrap();

                unsafe {
                    dequantize_per_tensor_symmetric_int8_packed_kernel::launch_unchecked::<F, R>(
                        &tensor.client,
                        cube_count,
                        cube_dim,
                        tensor.as_tensor_arg::<u32>(line_size_in),
                        scales.as_tensor_arg::<f32>(1),
                        output.as_tensor_arg::<F>(line_size_out),
                        scheme,
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
                    block_size % 4 == 0,
                    "block size must be divisible by 4, got block_size={block_size}"
                );
                let scales = tensor.scales().unwrap();

                unsafe {
                    dequantize_per_block_symmetric_int8_packed_kernel::launch_unchecked::<F, R>(
                        &tensor.client,
                        cube_count,
                        cube_dim,
                        tensor.as_tensor_arg::<u32>(line_size_in),
                        scales.as_tensor_arg::<f32>(1),
                        output.as_tensor_arg::<F>(line_size_out),
                        scheme,
                    )
                };
            }
        }
    }

    output
}

fn dequantize_native<R, F>(tensor: CubeTensor<R>, output: CubeTensor<R>) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
{
    let num_elems = tensor.shape.num_elements();
    let line_size = max_line_size(&tensor);
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    let out_layout = strided_layout(&output);

    if let DType::QFloat(scheme) = tensor.dtype {
        match scheme {
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                q_type: QuantInputType::QInt8,
                ..
            } => {
                let scales = tensor.scales().unwrap();

                unsafe {
                    dequantize_per_tensor_symmetric_int8_native_kernel::launch_unchecked::<F, R>(
                        &tensor.client,
                        cube_count,
                        cube_dim,
                        tensor.as_tensor_arg::<i8>(line_size),
                        scales.as_tensor_arg::<f32>(1),
                        output.as_tensor_arg::<F>(line_size),
                        out_layout,
                        scheme,
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
                    block_size as u8 % line_size == 0,
                    "block size must evenly divide line size, got {block_size} / {line_size}"
                );

                let scales = tensor.scales().unwrap();

                println!(
                    "dequantize per block native w/ scales {}",
                    crate::ops::into_data_sync::<R, f32>(scales.clone())
                );
                unsafe {
                    dequantize_per_block_symmetric_int8_native_kernel::launch_unchecked::<F, R>(
                        &tensor.client,
                        cube_count,
                        cube_dim,
                        tensor.as_tensor_arg::<i8>(line_size),
                        scales.as_tensor_arg::<f32>(1),
                        output.as_tensor_arg::<F>(line_size),
                        out_layout,
                        scheme,
                        Some(tensor.shape.num_dims() as u32),
                    )
                };
            }
        }
    }

    output
}
