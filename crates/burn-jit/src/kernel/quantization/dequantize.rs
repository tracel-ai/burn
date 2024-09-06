use crate::tensor::{JitTensor, QJitTensor};
use crate::FloatElement;
use crate::{kernel::Kernel, IntElement, JitElement, JitRuntime};
use burn_tensor::quantization::{QuantizationScheme, QuantizationType};
use cubecl::prelude::*;
use cubecl::{calculate_cube_count_elemwise, tensor_vectorization_factor};

#[cube]
pub(crate) fn dequantize_affine_int8<F: Float>(value: I32, scale: F, offset: I32) -> F {
    // x = scale * (x_q - offset)
    scale * (F::cast_from(value) - F::cast_from(offset))
}

#[cube]
pub(crate) fn extract_i8(value: UInt, offset: UInt) -> I32 {
    // Extract 8-bit segment
    let value = (value >> offset) & UInt::new(0xFF);
    // Check if the value is negative by inspecting the MSB and subtract 256 if it is
    // Subtract 0 or 256 to circumvent unsupported conditional assignment (let x = if {} else {};)
    let sub = I32::cast_from(value & UInt::new(0x80) != 0) * I32::new(256);
    I32::cast_from(value) - sub
}

#[cube(launch_unchecked)]
pub(crate) fn dequantize_per_tensor_affine_int8_kernel(
    input: &Tensor<UInt>,
    scale: &Tensor<F32>,
    offset: &Tensor<I32>,
    output: &mut Tensor<F32>,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let scale = scale[0];
    let offset = offset[0];

    let num_packed = UInt::new(4);
    let vectorization_factor = Comptime::vectorization(input);
    let vectorization = Comptime::get(vectorization_factor);

    let value = input[ABSOLUTE_POS];
    let output_pos = ABSOLUTE_POS * num_packed;

    for i in range(0u32, vectorization, Comptime::new(true)) {
        // Extract each 8-bit segment
        let v1 = extract_i8(value[i], UInt::new(24));
        let v2 = extract_i8(value[i], UInt::new(16));
        let v3 = extract_i8(value[i], UInt::new(8));
        let v4 = extract_i8(value[i], UInt::new(0));

        output[output_pos + i * num_packed] = dequantize_affine_int8::<F32>(v1, scale, offset);
        output[output_pos + i * num_packed + UInt::new(1)] =
            dequantize_affine_int8::<F32>(v2, scale, offset);
        output[output_pos + i * num_packed + UInt::new(2)] =
            dequantize_affine_int8::<F32>(v3, scale, offset);
        output[output_pos + i * num_packed + UInt::new(3)] =
            dequantize_affine_int8::<F32>(v4, scale, offset);
    }
}

#[cube]
pub(crate) fn dequantize_symmetric_int8<F: Float>(value: I32, scale: F) -> F {
    // x = scale * x_q
    scale * F::cast_from(value)
}

// Would have wrapped symmetric with the same affine kernel but cube doesn't support Option<Tensor> for offset.
#[cube(launch_unchecked)]
pub(crate) fn dequantize_per_tensor_symmetric_int8_kernel(
    input: &Tensor<UInt>,
    scale: &Tensor<F32>,
    output: &mut Tensor<F32>,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let scale = scale[0];

    let num_packed = UInt::new(4);
    let vectorization_factor = Comptime::vectorization(input);
    let vectorization = Comptime::get(vectorization_factor);

    let value = input[ABSOLUTE_POS];
    let output_pos = ABSOLUTE_POS * num_packed;

    for i in range(0u32, vectorization, Comptime::new(true)) {
        // Extract each 8-bit segment
        let v1 = extract_i8(value[i], UInt::new(24));
        let v2 = extract_i8(value[i], UInt::new(16));
        let v3 = extract_i8(value[i], UInt::new(8));
        let v4 = extract_i8(value[i], UInt::new(0));

        output[output_pos + i * num_packed] = dequantize_symmetric_int8::<F32>(v1, scale);
        output[output_pos + i * num_packed + UInt::new(1)] =
            dequantize_symmetric_int8::<F32>(v2, scale);
        output[output_pos + i * num_packed + UInt::new(2)] =
            dequantize_symmetric_int8::<F32>(v3, scale);
        output[output_pos + i * num_packed + UInt::new(3)] =
            dequantize_symmetric_int8::<F32>(v4, scale);
    }
}

pub(crate) fn dequantize_per_tensor<R, F, I, const D: usize>(
    tensor: JitTensor<R, u32, D>,
    scale: JitTensor<R, F, 1>,
    offset: Option<JitTensor<R, I, 1>>,
) -> JitTensor<R, F, D>
where
    R: JitRuntime,
    F: JitElement,
    I: IntElement,
{
    let num_elems = tensor.shape.num_elements();
    let shape_output = tensor.shape.clone();
    let client = tensor.client.clone();
    let handle = client.empty(num_elems * core::mem::size_of::<I>());
    let output =
        JitTensor::new_contiguous(client.clone(), tensor.device.clone(), shape_output, handle);

    // Vectorization is only enabled when the last dimension is contiguous.
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &tensor.shape.dims, &tensor.strides, D - 1);
    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    let dummy_array = [1; D];
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
                output.as_tensor_arg(vectorization_factor),
            )
        };
    }

    output
}

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R, F, I, const D: usize>(tensor: QJitTensor<R, F, I, D>) -> JitTensor<R, F, D>
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
