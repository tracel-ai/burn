use crate::tensor::CubeTensor;
use crate::FloatElement;
use crate::{CubeElement, CubeRuntime, IntElement};
use burn_tensor::quantization::{QuantizationScheme, QuantizationType};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

#[cube]
pub(crate) fn quantize_affine_int8<F: Float>(
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

#[cube(launch_unchecked)]
pub(crate) fn quantize_per_tensor_affine_int8_kernel(
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
        output[ABSOLUTE_POS] = u32::bitcast_from(scale);
        terminate!();
    }

    // Cast the offset to u32 and write the value in the output
    if ABSOLUTE_POS == output.len() - 2 {
        output[ABSOLUTE_POS] = u32::bitcast_from(offset);
        terminate!();
    }

    let line_size = comptime!(input.line_size());
    if comptime!(line_size == 4) {
        // Assuming a line size of 4 (equal to the number of values packed)
        let value =
            quantize_affine_int8::<f32>(input[ABSOLUTE_POS], scale, offset, range_min, range_max);
        // Shift and combine into u32
        output[ABSOLUTE_POS] = pack_i8s_to_u32s(value);
    } else {
        let mut v_packed = 0;
        let num_packed = comptime!(4);
        #[unroll]
        for i in 0..num_packed {
            let v = quantize_affine_int8::<f32>(
                input[ABSOLUTE_POS + i],
                scale,
                offset,
                range_min,
                range_max,
            );
            // Shift and combine into u32
            v_packed |= (v[0] & 0xFF) << (8 * i);
        }
        output[ABSOLUTE_POS] = v_packed;
    }
}

#[cube]
pub(crate) fn quantize_symmetric_int8<F: Float>(
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
pub(crate) fn pack_i8s_to_u32s(value: Line<u32>) -> u32 {
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

// Would have wrapped symmetric with the same affine kernel but cube doesn't support Option<Tensor> for offset.
#[cube(launch_unchecked)]
pub(crate) fn quantize_per_tensor_symmetric_int8_kernel(
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
        output[ABSOLUTE_POS] = u32::bitcast_from(scale);
        terminate!();
    }

    let line_size = comptime!(input.line_size());
    if comptime!(line_size == 4) {
        // Assuming a vectorization factor of 4 (equal to the number of values packed)
        let value =
            quantize_symmetric_int8::<f32>(input[ABSOLUTE_POS], scale, range_min, range_max);
        // Shift and combine into u32
        output[ABSOLUTE_POS] = pack_i8s_to_u32s(value);
    } else {
        let num_packed = comptime!(4);
        let mut v_packed = 0;
        #[unroll]
        for i in 0..num_packed {
            let v = quantize_symmetric_int8::<f32>(
                input[ABSOLUTE_POS + i],
                scale,
                range_min,
                range_max,
            );
            // Shift and combine into u32
            v_packed |= (v[0] & 0xFF) << (8 * i);
        }
        output[ABSOLUTE_POS] = v_packed;
    }
}

pub(crate) fn quantize_per_tensor<R, F, I>(
    tensor: CubeTensor<R>,
    scale: CubeTensor<R>,
    offset: Option<CubeTensor<R>>,
    scheme: QuantizationScheme,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: CubeElement,
    I: IntElement,
{
    let ndims = tensor.shape.num_dims();
    let num_elems = tensor.shape.num_elements();
    let client = tensor.client.clone();
    // Output tensor contains 4x less elements (four int8 values packed in a single u32)
    let output_num_elems = usize::div_ceil(num_elems, 4) * core::mem::size_of::<u32>();

    // Force vectorization to process 4 quantized values packed for 1 output value
    let line_size: u8 = if num_elems < 4 { 1 } else { 4 };
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    let dummy_array = vec![1; ndims];
    if let Some(offset) = offset {
        // Scale and offset qparams are also packed in the tensor dat
        let handle = client
            .empty(output_num_elems + core::mem::size_of::<f32>() + core::mem::size_of::<i32>());
        let output = CubeTensor::new_contiguous(
            client.clone(),
            tensor.device.clone(),
            tensor.shape.clone(),
            handle,
            burn_tensor::DType::QFloat(scheme),
        );

        unsafe {
            quantize_per_tensor_affine_int8_kernel::launch_unchecked::<R>(
                &client,
                cube_count,
                cube_dim,
                tensor.as_tensor_arg::<F>(line_size),
                // Ignore shape and stride
                TensorArg::from_raw_parts::<F>(&scale.handle, &dummy_array, &dummy_array, 1),
                TensorArg::from_raw_parts::<I>(&offset.handle, &dummy_array, &dummy_array, 1),
                ScalarArg::new(i8::MIN as f32),
                ScalarArg::new(i8::MAX as f32),
                output.as_array_arg::<u32>(1),
            )
        };
        output
    } else {
        // Scale qparam is also packed in the tensor data
        let handle = client.empty(output_num_elems + core::mem::size_of::<f32>());
        let output = CubeTensor::new_contiguous(
            client.clone(),
            tensor.device.clone(),
            tensor.shape.clone(),
            handle,
            burn_tensor::DType::QFloat(scheme),
        );

        unsafe {
            quantize_per_tensor_symmetric_int8_kernel::launch_unchecked::<R>(
                &client,
                cube_count,
                cube_dim,
                tensor.as_tensor_arg::<F>(line_size),
                // Ignore shape and stride
                TensorArg::from_raw_parts::<F>(&scale.handle, &dummy_array, &dummy_array, 1),
                ScalarArg::new(-i8::MAX as f32),
                ScalarArg::new(i8::MAX as f32),
                output.as_array_arg::<u32>(1),
            )
        };

        output
    }
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
    F: FloatElement,
    I: IntElement,
{
    match scheme {
        QuantizationScheme::PerTensor(_mode, QuantizationType::QInt8) => {
            quantize_per_tensor::<R, F, I>(tensor, scale, offset, *scheme)
        }
        QuantizationScheme::PerBlock(_mode, QuantizationType::QInt8, _block_layout) => todo!(),
    }
}
