#![allow(missing_docs)] // pub cube modules

use super::QParams;
use crate::{CubeRuntime, FloatElement, kernel::utils::strided_layout, ops::max_line_size};
use crate::{ops::numeric::empty_device_strided, tensor::CubeTensor};
use burn_tensor::quantization::{
    QuantFloatPrecision, QuantInputType, QuantLevel, QuantMode, QuantScheme, QuantStoreType,
};
use burn_tensor::{DType, bf16, f16};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::std::tensor::{StridedLayout, index_offset_contiguous};

/// Dequantize a line of values into floating-point values using the provided scale.
#[cube]
pub fn dequantize_symmetric<F: Float, FS: Float>(value: Line<F>, scale: FS) -> Line<F> {
    // x = scale * x_q
    Line::cast_from(scale) * value
}

// TODO: use for fusion @nath

/// Dequantize the value at a specified position using the provided quantization scheme.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_packed_values<F: Float, FS: Float, QI: Int>(
    position: u32,
    values: &Tensor<QI>,
    scales: &Tensor<FS>,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    let value = values[position];
    dequantize_packed_value_at::<F, FS, QI>(position, value, scales, scheme)
}

/// Dequantize a single value using the scale at the specified position.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_packed_value_at<F: Float, FS: Float, QI: Int>(
    position: u32,
    value: QI,
    scales: &Tensor<FS>,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    let qparams = QParams::new(scheme);
    let scale = qparams.scale(scales, position);
    dequantize_packed_value::<F, FS, QI>(value, scale, scheme)
}

/// Dequantize a single packed value using the scale provided.
///
/// Returns a line of floating-point values. The number of values in the line depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_packed_value<F: Float, FS: Float, QS: Int>(
    value: QS,
    scale: FS,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    // TODO: q_store_type: QuantStoreType::Native
    let floats = unpack_q::<F, QS>(value, scheme.q_type);

    dequantize_symmetric::<F, FS>(floats, scale)
}

/// Unpack a quantized integer into a line of floating-point values, according to the specified quantization input type.
///
/// This handles types where multiple quantized values are packed into a single integer (the stored quantization type).
#[allow(clippy::explicit_counter_loop)]
#[cube]
fn unpack_q<F: Float, QS: Int>(value: QS, #[comptime] quant: QuantInputType) -> Line<F> {
    let size_quant = comptime!(quant.size_bits() as u32);

    let size_store = comptime!(QS::size_bits().unwrap() as u32);
    let num_quant = comptime!(size_store / size_quant);

    let mut output = Line::empty(num_quant);
    let mut position = comptime!(0);

    let mask = QS::cast_from(comptime!((1 << size_quant) - 1));
    let sign_bit = QS::cast_from(comptime!(1 << (size_quant - 1)));
    let two_pow_n = comptime!(1 << size_quant);

    #[unroll]
    for _ in 0..num_quant {
        let offset = QS::cast_from(comptime!(position * size_quant));
        let raw = (value >> offset) & mask;

        // Branchless two's complement conversion
        // If raw >= 2^(n-1), then result = raw - 2^n
        let raw_i32 = i32::cast_from(raw);
        let is_negative = i32::cast_from(raw >= sign_bit); // 1 if negative, 0 if positive
        let signed_value = raw_i32 - (is_negative * two_pow_n);

        output[position] = F::cast_from(signed_value);
        comptime!(position += 1);
    }

    output
}

#[cube(launch_unchecked)]
fn dequantize_symmetric_packed_kernel<F: Float, FS: Float>(
    input: &Tensor<Line<u32>>,
    scales: &Tensor<FS>,
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

    let out = dequantize_packed_value::<F, FS, u32>(value, scale, scheme);

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
fn dequantize_symmetric_int8_native_kernel<F: Float, FS: Float>(
    input: &Tensor<Line<i8>>,
    scale: &Tensor<FS>,
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

    output[out_pos] = dequantize_symmetric::<F, FS>(Line::cast_from(input[in_pos]), scale);
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
            } => match scheme.q_params_precision {
                QuantFloatPrecision::F32 => dequantize_packed::<R, F, f32>(tensor, output),
                QuantFloatPrecision::F16 => dequantize_packed::<R, F, f16>(tensor, output),
                QuantFloatPrecision::BF16 => dequantize_packed::<R, F, bf16>(tensor, output),
            },
            QuantScheme {
                q_type: QuantInputType::QInt8,
                q_store_type: QuantStoreType::Native,
                ..
            } => {
                if !i8::is_supported(&tensor.client) {
                    panic!("QInt8 is not supported for native quantization");
                }

                match scheme.q_params_precision {
                    QuantFloatPrecision::F32 => dequantize_native::<R, F, f32>(tensor, output),
                    QuantFloatPrecision::F16 => dequantize_native::<R, F, f16>(tensor, output),
                    QuantFloatPrecision::BF16 => dequantize_native::<R, F, bf16>(tensor, output),
                }
            }
        },
        _ => panic!("Expected QFloat dtype"),
    }
}

fn dequantize_packed<R, F, FS>(tensor: CubeTensor<R>, output: CubeTensor<R>) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
    FS: FloatElement,
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
    let num_quants = (scheme.size_bits_stored() / scheme.q_type.size_bits()) as u8;
    let use_packed_line_size =
        num_out_elems % num_quants as usize == 0 && R::supported_line_sizes().contains(&num_quants);

    let line_size_out = if use_packed_line_size { num_quants } else { 1 };

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            q_store_type: QuantStoreType::U32,
            ..
        } => {
            super::check_block_size_compat(&scheme, num_quants as usize); // 32 / 8 = 4
            let scales = tensor.scales().unwrap();

            unsafe {
                dequantize_symmetric_packed_kernel::launch_unchecked::<F, FS, R>(
                    &tensor.client,
                    cube_count,
                    cube_dim,
                    tensor.as_tensor_arg::<u32>(line_size_in),
                    scales.as_tensor_arg::<FS>(1),
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

fn dequantize_native<R, F, FS>(tensor: CubeTensor<R>, output: CubeTensor<R>) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
    FS: FloatElement,
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
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            q_store_type: QuantStoreType::Native,
            ..
        } => {
            // We could use line_size = block_size if it's in the supported line sizes.. but let's keep it simple
            super::check_block_size_compat(&scheme, line_size as usize);
            let scales = tensor.scales().unwrap();

            unsafe {
                dequantize_symmetric_int8_native_kernel::launch_unchecked::<F, FS, R>(
                    &tensor.client,
                    cube_count,
                    cube_dim,
                    tensor.as_tensor_arg::<i8>(line_size),
                    scales.as_tensor_arg::<FS>(1),
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
