use crate::tensor::{JitQuantizationParameters, JitTensor, QJitTensor};
use crate::FloatElement;
use crate::{kernel::Kernel, IntElement, JitElement, JitRuntime};
use burn_tensor::quantization::{QuantizationScheme, QuantizationType};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

#[cube]
pub(crate) fn quantize_affine_int8<F: Float>(
    value: F,
    scale: F,
    offset: I32,
    range_min: F,
    range_max: F,
) -> UInt {
    let offset = F::cast_from(offset);

    // x_q = clamp(round(x / scale + offset), a, b)
    UInt::cast_from(F::clamp(
        F::round((value / scale) + offset),
        range_min,
        range_max,
    ))
}

#[cube(launch_unchecked)]
pub(crate) fn quantize_per_tensor_affine_int8_kernel(
    input: &Tensor<F32>,
    scale: &Tensor<F32>,
    offset: &Tensor<I32>,
    range_min: F32,
    range_max: F32,
    output: &mut Tensor<UInt>,
    vectorized: Comptime<bool>,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let scale = scale[0];
    let offset = offset[0];

    // Assuming a vectorization factor of 4 (equal to the number of values packed)
    let num_packed = UInt::new(4);
    let vectorization_factor = Comptime::vectorization(input);
    let vectorization = Comptime::get(vectorization_factor);

    let bit_shift = UInt::new(8);
    let mut v_packed = UInt::new(0);

    if Comptime::get(vectorized) {
        let value = input[ABSOLUTE_POS];
        for i in range(0u32, vectorization, Comptime::new(true)) {
            let v = quantize_affine_int8::<F32>(value[i], scale, offset, range_min, range_max);
            // Shift and combine into u32
            v_packed = v_packed | (v & UInt::new(0xFF)) << (bit_shift * (num_packed - i - 1));
        }
    } else {
        for i in range(0u32, num_packed, Comptime::new(false)) {
            let v = quantize_affine_int8::<F32>(
                input[ABSOLUTE_POS + i],
                scale,
                offset,
                range_min,
                range_max,
            );
            // Shift and combine into u32
            v_packed = v_packed | (v & UInt::new(0xFF)) << (bit_shift * (num_packed - i - 1));
        }
    }

    output[ABSOLUTE_POS] = v_packed;
}

#[cube]
pub(crate) fn quantize_symmetric_int8<F: Float>(
    value: F,
    scale: F,
    range_min: F,
    range_max: F,
) -> UInt {
    // x_q = clamp(round(x / scale), a, b)
    UInt::cast_from(F::clamp(F::round(value / scale), range_min, range_max))
}

// Would have wrapped symmetric with the same affine kernel but cube doesn't support Option<Tensor> for offset.
#[cube(launch_unchecked)]
pub(crate) fn quantize_per_tensor_symmetric_int8_kernel(
    input: &Tensor<F32>,
    scale: &Tensor<F32>,
    range_min: F32,
    range_max: F32,
    output: &mut Tensor<UInt>,
    vectorized: Comptime<bool>,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let scale = scale[0];

    let num_packed = UInt::new(4);
    let bit_shift = UInt::new(8);
    let mut v_packed = UInt::new(0);

    if Comptime::get(vectorized) {
        // Assuming a vectorization factor of 4 (equal to the number of values packed)
        let value = input[ABSOLUTE_POS];
        let vectorization_factor = Comptime::vectorization(input);
        let vectorization = Comptime::get(vectorization_factor);
        for i in range(0u32, vectorization, Comptime::new(true)) {
            let v = quantize_symmetric_int8::<F32>(value[i], scale, range_min, range_max);
            // Shift and combine into u32
            v_packed = v_packed | (v & UInt::new(0xFF)) << (bit_shift * (num_packed - i - 1));
        }
    } else {
        for i in range(0u32, num_packed, Comptime::new(false)) {
            let v = quantize_symmetric_int8::<F32>(
                input[ABSOLUTE_POS + i],
                scale,
                range_min,
                range_max,
            );
            // Shift and combine into u32
            v_packed = v_packed | (v & UInt::new(0xFF)) << (bit_shift * (num_packed - i - 1));
        }
    }

    output[ABSOLUTE_POS] = v_packed;
}

pub(crate) fn quantize_per_tensor<R, F, I, const D: usize>(
    tensor: JitTensor<R, F, D>,
    scale: JitTensor<R, F, 1>,
    offset: Option<JitTensor<R, I, 1>>,
) -> JitTensor<R, u32, D>
where
    R: JitRuntime,
    F: JitElement,
    I: IntElement,
{
    let num_elems = tensor.shape.num_elements();
    let shape_output = tensor.shape.clone();
    let client = tensor.client.clone();
    // Output tensor contains 4x less elements (four int8 values packed in a single u32)
    let handle = client.empty(usize::div_ceil(num_elems, 4) * core::mem::size_of::<I>());
    let output =
        JitTensor::new_contiguous(client.clone(), tensor.device.clone(), shape_output, handle);

    // Force vectorization to process 4 quantized values packed for 1 output value
    let vectorization_factor: u8 = if num_elems < 4 { 1 } else { 4 };
    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    let dummy_array = [1; D];
    if let Some(offset) = offset {
        unsafe {
            quantize_per_tensor_affine_int8_kernel::launch_unchecked::<R>(
                &client,
                cube_count,
                cube_dim,
                tensor.as_tensor_arg(vectorization_factor),
                // Ignore shape and stride
                TensorArg::from_raw_parts(&scale.handle, &dummy_array, &dummy_array, 1),
                TensorArg::from_raw_parts(&offset.handle, &dummy_array, &dummy_array, 1),
                ScalarArg::new(i8::MIN as f32),
                ScalarArg::new(i8::MAX as f32),
                output.as_tensor_arg(1),
                vectorization_factor > 1,
            )
        };
    } else {
        unsafe {
            quantize_per_tensor_symmetric_int8_kernel::launch_unchecked::<R>(
                &client,
                cube_count,
                cube_dim,
                tensor.as_tensor_arg(vectorization_factor),
                // Ignore shape and stride
                TensorArg::from_raw_parts(&scale.handle, &dummy_array, &dummy_array, 1),
                ScalarArg::new(-i8::MAX as f32),
                ScalarArg::new(i8::MAX as f32),
                output.as_tensor_arg(1),
                vectorization_factor > 1,
            )
        };
    }

    output
}

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R, F, I, const D: usize>(
    tensor: JitTensor<R, F, D>,
    scheme: &QuantizationScheme,
    qparams: JitQuantizationParameters<R, F, I>,
) -> QJitTensor<R, F, I, D>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
    let qtensor = match scheme {
        QuantizationScheme::PerTensorAffine(dtype)
        | QuantizationScheme::PerTensorSymmetric(dtype) => match dtype {
            QuantizationType::QInt8 => {
                quantize_per_tensor(tensor, qparams.scale.clone(), qparams.offset.clone())
            }
        },
    };

    QJitTensor {
        qtensor,
        scheme: scheme.clone(),
        qparams,
    }
}
