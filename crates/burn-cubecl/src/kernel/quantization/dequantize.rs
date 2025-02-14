use crate::tensor::CubeTensor;
use crate::FloatElement;
use crate::{CubeElement, CubeRuntime};
use burn_tensor::quantization::{QuantizationMode, QuantizationScheme, QuantizationType};
use burn_tensor::DType;
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

use super::{QParams, QTensor};

#[cube]
pub(crate) fn dequantize_affine_int8<F: Float>(
    value: Line<i32>,
    scale: f32,
    offset: i32,
) -> Line<F> {
    // x = scale * (x_q - offset)
    Line::cast_from(scale) * Line::cast_from(value - Line::cast_from(offset))
}

#[cube]
pub(crate) fn extract_i8(value: u32, offset: u32) -> i32 {
    // Extract 8-bit segment
    let value = (value >> offset) & 0xFF;
    // Check if the value is negative by inspecting the MSB and subtract 256 if it is
    // Subtract 0 or 256 to circumvent unsupported conditional assignment (let x = if {} else {};)
    let sub = i32::cast_from(value & 0x80 != 0) * 256;
    i32::cast_from(value) - sub
}

#[cube]
pub(crate) fn extract_i8s(value: u32) -> Line<i32> {
    let mut line = Line::empty(4);
    // Extract each 8-bit segment
    line[0] = extract_i8(value, 0);
    line[1] = extract_i8(value, 8);
    line[2] = extract_i8(value, 16);
    line[3] = extract_i8(value, 24);

    line
}

#[cube(launch_unchecked)]
pub(crate) fn dequantize_per_tensor_affine_int8_kernel(
    input: &QTensor,
    output: &mut Tensor<Line<f32>>,
    #[comptime] scheme: QuantizationScheme,
) {
    // Last two positions contain the qparams
    if ABSOLUTE_POS >= input.len() - 2 {
        terminate!();
    }

    let qparams = QParams::new(scheme);
    let (scale, offset) = qparams.values(input);

    let value = input[ABSOLUTE_POS];

    // Input line size is fixed to 1
    if comptime!(output.line_size() == 4) {
        output[ABSOLUTE_POS] = dequantize_affine_int8(extract_i8s(value[0]), scale, offset);
    } else {
        // For very small inputs where number of elements < 4, the output line size is 1
        let out = dequantize_affine_int8::<f32>(extract_i8s(value[0]), scale, offset);

        #[unroll]
        for j in 0..out.size() {
            output[ABSOLUTE_POS + j] = Line::cast_from(out[j]);
        }
    }
}

#[cube]
pub(crate) fn dequantize_symmetric_int8<F: Float>(value: Line<i32>, scale: f32) -> Line<F> {
    // x = scale * x_q
    Line::cast_from(scale) * Line::cast_from(value)
}

// Would have wrapped symmetric with the same affine kernel but cube doesn't support Option<Tensor> for offset.
#[cube(launch_unchecked)]
pub(crate) fn dequantize_per_tensor_symmetric_int8_kernel(
    input: &QTensor,
    output: &mut Tensor<Line<f32>>,
    #[comptime] scheme: QuantizationScheme,
) {
    // Last position contains the qparam
    if ABSOLUTE_POS >= input.len() - 1 {
        terminate!();
    }

    let qparams = QParams::new(scheme);
    let (scale, _) = qparams.values(input);

    let value = input[ABSOLUTE_POS];

    // Input line size is fixed to 1
    if comptime!(output.line_size() == 4) {
        output[ABSOLUTE_POS] = dequantize_symmetric_int8(extract_i8s(value[0]), scale);
    } else {
        // For very small inputs where number of elements < 4, the output line size is 1
        let out = dequantize_symmetric_int8::<f32>(extract_i8s(value[0]), scale);

        #[unroll]
        for j in 0..out.size() {
            output[ABSOLUTE_POS + j] = Line::cast_from(out[j]);
        }
    }
}

pub(crate) fn dequantize_per_tensor<R, F>(tensor: CubeTensor<R>) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: CubeElement,
{
    // The actual number of elements is 1/4 (four int8 values packed in a single u32)
    // so we choose a line size to match a valid input binding size.
    let num_out_elems = tensor.shape.num_elements();
    let num_elems = usize::div_ceil(num_out_elems, 4);
    let line_size_in = 1;
    let line_size_out = if num_out_elems < 4 { 1 } else { 4 };
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size_in as usize, cube_dim);

    let client = tensor.client.clone();
    let handle = client.empty(num_out_elems * core::mem::size_of::<F>());

    let output = CubeTensor::new_contiguous(
        client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
        handle,
        F::dtype(),
    );

    if let DType::QFloat(scheme) = tensor.dtype {
        match scheme {
            QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8) => {
                unsafe {
                    dequantize_per_tensor_affine_int8_kernel::launch_unchecked::<R>(
                        &client,
                        cube_count,
                        cube_dim,
                        tensor.as_array_arg::<u32>(line_size_in),
                        output.as_tensor_arg::<F>(line_size_out),
                        scheme,
                    )
                };
            }
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8) => {
                unsafe {
                    dequantize_per_tensor_symmetric_int8_kernel::launch_unchecked::<R>(
                        &client,
                        cube_count,
                        cube_dim,
                        tensor.as_array_arg::<u32>(line_size_in),
                        output.as_tensor_arg::<F>(line_size_out),
                        scheme,
                    )
                };
            }
            QuantizationScheme::PerBlock(_mode, QuantizationType::QInt8, _block_layout) => todo!(),
        }
    }

    output
}

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R, F>(tensor: CubeTensor<R>) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
{
    dequantize_per_tensor::<R, F>(tensor)
}
