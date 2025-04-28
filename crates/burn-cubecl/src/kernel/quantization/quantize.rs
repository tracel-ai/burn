use crate::tensor::CubeTensor;
use crate::{CubeElement, CubeRuntime, IntElement};
use burn_tensor::Shape;
use burn_tensor::quantization::{QuantInputType, QuantLevel, QuantMode, QuantScheme};
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
            values[i] = input[ABSOLUTE_POS * num_packed + i][0];
        }
        output[ABSOLUTE_POS] = quantize_symmetric_int8_packed(values, scale, range_min, range_max);
    }
}

fn create_quantized_output<R: CubeRuntime>(
    client: ComputeClient<R::Server, R::Channel>,
    num_input_elems: usize,
    device: R::Device,
    shape: Shape,
    scheme: QuantScheme,
) -> CubeTensor<R> {
    // Output tensor contains 4x less elements (four int8 values packed in a single u32)
    let output_elems_size = usize::div_ceil(num_input_elems, 4) * core::mem::size_of::<u32>();

    // Scale and offset (optional) qparams are also packed in the tensor data
    let qparams_size = match &scheme {
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            acc_precision: _,
            propagation: _,
        } => core::mem::size_of::<f32>(),
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
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
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
    let line_size: u8 = 1;
    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems.div_ceil(line_size as usize), cube_dim);

    let output = create_quantized_output(
        client.clone(),
        num_elems,
        tensor.device.clone(),
        tensor.shape.clone(),
        *scheme,
    );

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            acc_precision: _,
            propagation: _,
        } => {
            let ndims = tensor.shape.num_dims();
            let dummy_array = vec![1; ndims];

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
        }
    }

    output
}
