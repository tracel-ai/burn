use crate::tensor::{JitTensor, QJitTensor};
use crate::FloatElement;
use crate::{IntElement, JitElement, JitRuntime};
use burn_tensor::quantization::{QuantizationScheme, QuantizationType};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

#[cube]
pub(crate) fn dequantize_affine_int8<F: Float>(value: i32, scale: F, offset: i32) -> F {
    // x = scale * (x_q - offset)
    scale * (F::cast_from(value) - F::cast_from(offset))
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

#[cube(launch_unchecked)]
pub(crate) fn dequantize_per_tensor_affine_int8_kernel(
    input: &Tensor<u32>,
    scale: &Tensor<f32>,
    offset: &Tensor<i32>,
    output: &mut Tensor<f32>,
    #[comptime] vectorized: bool,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let scale = scale[0];
    let offset = offset[0];

    let num_packed = 4;
    let value = input[ABSOLUTE_POS];
    let output_pos = ABSOLUTE_POS * num_packed;

    if vectorized {
        let vectorization_factor = vectorization_of(input);
        #[unroll]
        for i in 0..vectorization_factor {
            // Extract each 8-bit segment
            let v1 = extract_i8(value[i], 24);
            let v2 = extract_i8(value[i], 16);
            let v3 = extract_i8(value[i], 8);
            let v4 = extract_i8(value[i], 0);

            output[output_pos * vectorization_factor + i * num_packed] =
                dequantize_affine_int8::<f32>(v1, scale, offset);
            output[output_pos * vectorization_factor + i * num_packed + 1] =
                dequantize_affine_int8::<f32>(v2, scale, offset);
            output[output_pos * vectorization_factor + i * num_packed + 2] =
                dequantize_affine_int8::<f32>(v3, scale, offset);
            output[output_pos * vectorization_factor + i * num_packed + 3] =
                dequantize_affine_int8::<f32>(v4, scale, offset);
        }
    } else {
        // Extract each 8-bit segment
        let v1 = extract_i8(value, 24);
        let v2 = extract_i8(value, 16);
        let v3 = extract_i8(value, 8);
        let v4 = extract_i8(value, 0);

        output[output_pos] = dequantize_affine_int8::<f32>(v1, scale, offset);
        output[output_pos + 1] = dequantize_affine_int8::<f32>(v2, scale, offset);
        output[output_pos + 2] = dequantize_affine_int8::<f32>(v3, scale, offset);
        output[output_pos + 3] = dequantize_affine_int8::<f32>(v4, scale, offset);
    }
}

#[cube]
pub(crate) fn dequantize_symmetric_int8<F: Float>(value: i32, scale: F) -> F {
    // x = scale * x_q
    scale * F::cast_from(value)
}

// Would have wrapped symmetric with the same affine kernel but cube doesn't support Option<Tensor> for offset.
#[cube(launch_unchecked)]
pub(crate) fn dequantize_per_tensor_symmetric_int8_kernel(
    input: &Tensor<u32>,
    scale: &Tensor<f32>,
    output: &mut Tensor<f32>,
    #[comptime] vectorized: bool,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let scale = scale[0];

    let num_packed = 4;
    let value = input[ABSOLUTE_POS];
    let output_pos = ABSOLUTE_POS * num_packed;

    if vectorized {
        let vectorization_factor = vectorization_of(input);
        #[unroll]
        for i in 0..vectorization_factor {
            for j in 0..num_packed {
                let output_idx = output_pos * vectorization_factor + i * num_packed + j;
                if output_idx >= output.len() {
                    return; // value not quantized (padding)
                }
                // Extract each 8-bit segment
                let v = extract_i8(value[i], (3 - j) * 8);
                output[output_idx] = dequantize_symmetric_int8::<f32>(v, scale);
            }
        }
    } else {
        // Extract each 8-bit segment
        for j in 0..num_packed {
            let output_idx = output_pos + j;
            if output_idx >= output.len() {
                return; // value not quantized (padding)
            }
            // Extract each 8-bit segment
            let v = extract_i8(value, (3 - j) * 8);
            output[output_pos + j] = dequantize_symmetric_int8::<f32>(v, scale);
        }
    }
}

pub(crate) fn dequantize_per_tensor<R, F, I>(
    tensor: JitTensor<R, u32>,
    scale: JitTensor<R, F>,
    offset: Option<JitTensor<R, I>>,
) -> JitTensor<R, F>
where
    R: JitRuntime,
    F: JitElement,
    I: IntElement,
{
    // The actual number of elements is 1/4 (four int8 values packed in a single u32)
    // so we choose a vectorization factor to match a valid input binding size.
    let ndims = tensor.shape.num_dims();
    let num_out_elems = tensor.shape.num_elements();
    let num_elems = usize::div_ceil(num_out_elems, 4);
    let vectorization_factor = [4u8, 2, 1]
        .iter()
        .filter_map(|&v| {
            if num_elems >= v as usize {
                Some(v)
            } else {
                None
            }
        })
        .next()
        .unwrap();
    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    let shape_output = tensor.shape.clone();
    let client = tensor.client.clone();
    let handle = client.empty(num_out_elems * core::mem::size_of::<F>());
    let output =
        JitTensor::new_contiguous(client.clone(), tensor.device.clone(), shape_output, handle);

    let dummy_array = vec![1; ndims];
    if let Some(offset) = offset {
        unsafe {
            dequantize_per_tensor_affine_int8_kernel::launch_unchecked::<R>(
                &client,
                cube_count,
                cube_dim,
                tensor.as_tensor_arg(vectorization_factor),
                // Ignore shape and stride
                TensorArg::from_raw_parts(&scale.handle, &dummy_array, &dummy_array, 1),
                TensorArg::from_raw_parts(&offset.handle, &dummy_array, &dummy_array, 1),
                output.as_tensor_arg(1),
                vectorization_factor > 1,
            )
        };
    } else {
        unsafe {
            dequantize_per_tensor_symmetric_int8_kernel::launch_unchecked::<R>(
                &client,
                cube_count,
                cube_dim,
                tensor.as_tensor_arg(vectorization_factor),
                // Ignore shape and stride
                TensorArg::from_raw_parts(&scale.handle, &dummy_array, &dummy_array, 1),
                output.as_tensor_arg(1),
                vectorization_factor > 1,
            )
        };
    }

    output
}

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R, F, I>(tensor: QJitTensor<R, F, I>) -> JitTensor<R, F>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
    match tensor.scheme {
        QuantizationScheme::PerTensorAffine(dtype)
        | QuantizationScheme::PerTensorSymmetric(dtype) => match dtype {
            QuantizationType::QInt8 => {
                dequantize_per_tensor(tensor.qtensor, tensor.qparams.scale, tensor.qparams.offset)
            }
        },
    }
}
