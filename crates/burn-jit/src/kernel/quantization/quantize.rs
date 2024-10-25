use crate::tensor::{JitQuantizationParameters, JitTensor, QJitTensor};
use crate::FloatElement;
use crate::{IntElement, JitElement, JitRuntime};
use burn_tensor::quantization::{QuantizationScheme, QuantizationType};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

#[cube]
pub(crate) fn quantize_affine_int8<F: Float>(
    value: F,
    scale: F,
    offset: i32,
    range_min: F,
    range_max: F,
) -> u32 {
    let offset = F::cast_from(offset);

    // x_q = clamp(round(x / scale + offset), a, b)
    // NOTE: we add 256 before casting to unsigned to correctly represent negative values
    u32::cast_from(
        i32::cast_from(F::clamp(
            F::round((value / scale) + offset),
            range_min,
            range_max,
        )) + 256,
    )
}

#[cube(launch_unchecked)]
pub(crate) fn quantize_per_tensor_affine_int8_kernel(
    input: &Tensor<f32>,
    scale: &Tensor<f32>,
    offset: &Tensor<i32>,
    range_min: f32,
    range_max: f32,
    output: &mut Tensor<u32>,
    #[comptime] vectorized: bool,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let scale = scale[0];
    let offset = offset[0];

    let num_packed = 4;
    let mut v_packed = 0;

    if vectorized {
        // Assuming a vectorization factor of 4 (equal to the number of values packed)
        let value = input[ABSOLUTE_POS];
        let vectorization_factor = vectorization_of(input);
        #[unroll]
        for i in 0..vectorization_factor {
            let v = quantize_affine_int8::<f32>(value[i], scale, offset, range_min, range_max);
            // Shift and combine into u32
            v_packed |= (v & 0xFF) << (8 * (num_packed - i - 1));
        }
    } else {
        for i in 0..num_packed {
            let v = quantize_affine_int8::<f32>(
                input[ABSOLUTE_POS + i],
                scale,
                offset,
                range_min,
                range_max,
            );
            // Shift and combine into u32
            v_packed |= (v & 0xFF) << (8 * (num_packed - i - 1));
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
) -> u32 {
    // x_q = clamp(round(x / scale), a, b)
    // NOTE: we add 256 before casting to unsigned to correctly represent negative values
    u32::cast_from(i32::cast_from(F::clamp(F::round(value / scale), range_min, range_max)) + 256)
}

// Would have wrapped symmetric with the same affine kernel but cube doesn't support Option<Tensor> for offset.
#[cube(launch_unchecked)]
pub(crate) fn quantize_per_tensor_symmetric_int8_kernel(
    input: &Tensor<f32>,
    scale: &Tensor<f32>,
    range_min: f32,
    range_max: f32,
    output: &mut Tensor<u32>,
    #[comptime] vectorized: bool,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let scale = scale[0];

    let num_packed = 4;
    let mut v_packed = 0;

    if vectorized {
        // Assuming a vectorization factor of 4 (equal to the number of values packed)
        let value = input[ABSOLUTE_POS];
        let vectorization_factor = vectorization_of(input);
        #[unroll]
        for i in 0..vectorization_factor {
            let v = quantize_symmetric_int8::<f32>(value[i], scale, range_min, range_max);
            // Shift and combine into u32
            v_packed |= (v & 0xFF) << (8 * (num_packed - i - 1));
        }
    } else {
        for i in 0..num_packed {
            let v = quantize_symmetric_int8::<f32>(
                input[ABSOLUTE_POS + i],
                scale,
                range_min,
                range_max,
            );
            // Shift and combine into u32
            v_packed |= (v & 0xFF) << (8 * (num_packed - i - 1));
        }
    }

    output[ABSOLUTE_POS] = v_packed;
}

pub(crate) fn quantize_per_tensor<R, F, I>(
    tensor: JitTensor<R, F>,
    scale: JitTensor<R, F>,
    offset: Option<JitTensor<R, I>>,
) -> JitTensor<R, u32>
where
    R: JitRuntime,
    F: JitElement,
    I: IntElement,
{
    let ndims = tensor.shape.num_dims();
    let num_elems = tensor.shape.num_elements();
    let shape_output = tensor.shape.clone();
    let client = tensor.client.clone();
    // Output tensor contains 4x less elements (four int8 values packed in a single u32)
    let handle = client.empty(usize::div_ceil(num_elems, 4) * core::mem::size_of::<u32>());
    let output =
        JitTensor::new_contiguous(client.clone(), tensor.device.clone(), shape_output, handle);

    // Force vectorization to process 4 quantized values packed for 1 output value
    let vectorization_factor: u8 = if num_elems < 4 { 1 } else { 4 };
    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    let dummy_array = vec![1; ndims];
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
pub fn quantize<R, F, I>(
    tensor: JitTensor<R, F>,
    scheme: &QuantizationScheme,
    qparams: JitQuantizationParameters<R, F, I>,
) -> QJitTensor<R, F, I>
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
