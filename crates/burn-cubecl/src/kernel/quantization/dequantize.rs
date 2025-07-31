#![allow(missing_docs)] // pub cube modules

use super::QParams;
use crate::{CubeRuntime, FloatElement, kernel::utils::strided_layout, ops::max_line_size};
use crate::{ops::numeric::empty_device_strided, tensor::CubeTensor};
use burn_tensor::DType;
use burn_tensor::quantization::{
    QuantInputType, QuantLevel, QuantMode, QuantScheme, QuantStoreType,
};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::std::tensor::{StridedLayout, index_offset_contiguous};

/// Dequantize the value at a specified position using the provided quantization scheme.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_packed_values<F: Float, QI: Int>(
    position: u32,
    values: &Tensor<QI>,
    scales: &Tensor<f32>,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    let value = values[position];
    dequantize_packed_value_at::<F, QI>(position, value, scales, scheme)
}

/// Dequantize a single value using the scale at the specified position.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_packed_value_at<F: Float, QI: Int>(
    position: u32,
    value: QI,
    scales: &Tensor<f32>,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    let qparams = QParams::new(scheme);
    let scale = qparams.scale(scales, position);
    dequantize_packed_value::<F, QI>(value, scale, scheme)
}

/// Dequantize a single packed value using the scale provided.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_packed_value<F: Float, QI: Int>(
    value: QI,
    scale: f32,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    // TODO: q_store_type: QuantStoreType::Native
    let floats = unpack_q::<F, QI>(value, scheme.q_type);

    dequantize_symmetric(floats, scale)
}

/// Dequantize a line of values into floating-point values using the provided scale.
#[cube]
pub fn dequantize_symmetric<F: Float>(value: Line<F>, scale: f32) -> Line<F> {
    // x = scale * x_q
    Line::cast_from(scale) * value
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

    let mut output = Line::empty(num_quant);
    let mut position = comptime!(0);
    let mask = QI::cast_from(comptime!((1 << size_quant) - 1));
    let shift_sign = QI::cast_from(comptime!(24));

    #[unroll]
    for _ in 0..num_quant {
        let offset = QI::cast_from(comptime!(position * size_quant));
        let raw = (value >> offset) & mask;
        // Sign-extend: move sign bit to MSB via leftshift, then rightshift to restore sign
        output[position] = F::cast_from(i32::cast_from(raw << shift_sign) >> 24);
        comptime!(position += 1);
    }

    output
}

#[cube(launch_unchecked)]
fn dequantize_symmetric_packed_kernel<F: Float>(
    input: &Tensor<Line<u32>>,
    scales: &Tensor<f32>,
    output: &mut Tensor<Line<F>>,
    #[comptime] scheme: QuantScheme,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    // Input line size = 1
    let qparams = QParams::new(scheme);
    let num_quants = comptime!(qparams.num_quants);
    let scale = qparams.scale(scales, ABSOLUTE_POS);
    let value = input[ABSOLUTE_POS][0];

    let out = dequantize_packed_value::<F, u32>(value, scale, scheme);

    if comptime!(output.line_size() == num_quants) {
        output[ABSOLUTE_POS] = out;
    } else {
        // Output line size = 1
        #[unroll]
        for i in 0..out.size() {
            output[ABSOLUTE_POS * out.size() + i] = Line::cast_from(out[i]);
        }
    }
}

#[cube(launch_unchecked)]
fn dequantize_symmetric_int8_native_kernel<F: Float>(
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

    output[out_pos] = dequantize_symmetric(Line::cast_from(input[in_pos]), scale);
}

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R, F>(tensor: CubeTensor<R>) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
{
    let shape = tensor.shape.clone();
    let output = empty_device_strided::<R, F>(tensor.client.clone(), tensor.device.clone(), shape);

    match tensor.dtype {
        DType::QFloat(scheme) => match scheme {
            QuantScheme {
                q_type: QuantInputType::QInt8,
                q_store_type: QuantStoreType::U32,
                ..
            } => dequantize_packed::<R, F>(tensor, output),
            QuantScheme {
                q_type: QuantInputType::QInt8,
                q_store_type: QuantStoreType::Native,
                ..
            } => {
                if !i8::is_supported(&tensor.client) {
                    panic!("QInt8 is not supported for native quantization");
                }

                dequantize_native::<R, F>(tensor, output)
            }
        },
        _ => panic!("Expected QFloat dtype"),
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
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size_in as usize, cube_dim);

    let scheme = match tensor.dtype {
        DType::QFloat(s) => s,
        _ => panic!("Expected QFloat dtype"),
    };

    // Output line size selected based on the number of packed values per storage type
    let num_quants = (scheme.bits_stored() / scheme.bits_type()) as u8;
    let use_packed_line_size =
        num_out_elems % num_quants as usize == 0 && R::supported_line_sizes().contains(&num_quants);

    let line_size_out = if use_packed_line_size { num_quants } else { 1 };

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            q_store_type: QuantStoreType::U32,
            ..
        } => {
            let scales = tensor.scales().unwrap();

            unsafe {
                dequantize_symmetric_packed_kernel::launch_unchecked::<F, R>(
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
            q_store_type: QuantStoreType::U32,
            ..
        } => {
            assert!(
                block_size % 4 == 0,
                "Block size must be divisible by 4, got block_size={block_size}"
            );
            let scales = tensor.scales().unwrap();

            unsafe {
                dequantize_symmetric_packed_kernel::launch_unchecked::<F, R>(
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
            q_store_type: QuantStoreType::Native,
            ..
        } => panic!("Invalid quantization storage type for scheme {scheme:?}"),
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

    let scheme = match tensor.dtype {
        DType::QFloat(s) => s,
        _ => panic!("Expected QFloat dtype"),
    };

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            q_store_type: QuantStoreType::Native,
            ..
        } => {
            let scales = tensor.scales().unwrap();

            unsafe {
                dequantize_symmetric_int8_native_kernel::launch_unchecked::<F, R>(
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
            q_store_type: QuantStoreType::Native,
            ..
        } => {
            // We could use line_size = block_size if it's in the supported line sizes.. but let's keep it simple
            assert!(
                block_size as u8 % line_size == 0,
                "Block size must evenly divide line size, got {block_size} / {line_size}"
            );

            let scales = tensor.scales().unwrap();

            unsafe {
                dequantize_symmetric_int8_native_kernel::launch_unchecked::<F, R>(
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
            q_store_type: QuantStoreType::U32,
            ..
        } => panic!("Invalid quantization storage type for scheme {scheme:?}"),
    }

    output
}
