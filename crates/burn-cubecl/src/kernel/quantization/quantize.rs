use crate::{
    CubeRuntime, FloatElement,
    kernel::into_contiguous,
    ops::{into_data_sync, max_line_size},
};
use crate::{kernel::utils::strided_layout, tensor::CubeTensor};
use burn_tensor::Shape;
use burn_tensor::quantization::{QuantInputType, QuantLevel, QuantMode, QuantScheme};
use cubecl::linalg::tensor::index_offset_contiguous;
use cubecl::prelude::*;
use cubecl::{calculate_cube_count_elemwise, linalg::tensor::StridedLayout};

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
) -> Line<i8> {
    // x_q = clamp(round(x / scale), a, b)
    Line::cast_from(Line::clamp(
        Line::round(value / Line::cast_from(scale)),
        Line::new(range_min),
        Line::new(range_max),
    ))
}

#[cube]
fn quantize_symmetric_int8_packed<F: Float>(
    input: Line<F>,
    scale: f32,
    range_min: F,
    range_max: F,
) -> u32 {
    // Assuming a line size of 4 (equal to the number of values packed)
    // x_q = clamp(round(x / scale), a, b)
    // NOTE: we add 256 before casting to unsigned to correctly represent negative values
    let value = Line::cast_from(
        Line::clamp(
            Line::round(input / Line::cast_from(scale)),
            Line::new(range_min),
            Line::new(range_max),
        ) + Line::cast_from(comptime!(256f32)),
    );
    // Shift and combine into u32
    pack_i8s_to_u32s(value)
}

#[cube(launch_unchecked)]
fn quantize_per_tensor_symmetric_int8_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    scale: &Array<f32>,
    range_min: F,
    range_max: F,
    output: &mut Tensor<Line<i8>>,
    out_scale: &mut Array<f32>,
    out_layout: StridedLayout,
    #[comptime] rank: Option<u32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let scale = scale[0];

    // Write the scale into the output buffer
    if ABSOLUTE_POS == 0 {
        out_scale[ABSOLUTE_POS] = scale;
    }

    let in_pos = index_offset_contiguous(input, ABSOLUTE_POS, rank);
    let out_pos = out_layout.index(output, ABSOLUTE_POS);

    output[out_pos] = quantize_symmetric_int8(input[in_pos], scale, range_min, range_max);
}

#[cube(launch_unchecked)]
fn quantize_per_tensor_symmetric_int8_packed_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    scale: &Array<f32>,
    range_min: F,
    range_max: F,
    output: &mut Array<u32>,
    out_scale: &mut Array<f32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let scale = scale[0];

    if ABSOLUTE_POS == 0 {
        out_scale[0] = scale;
        let out_scale = out_scale[0];
        debug_print!("Wrote scale: %f", out_scale);
    }

    if comptime!(input.line_size() == 4) {
        output[ABSOLUTE_POS] =
            quantize_symmetric_int8_packed::<F>(input[ABSOLUTE_POS], scale, range_min, range_max);
    } else {
        // line size 1
        let num_packed = comptime!(4);
        let mut values = Line::<F>::empty(num_packed);
        #[unroll]
        for i in 0..num_packed {
            values[i] = input[ABSOLUTE_POS * num_packed + i][0];
        }
        output[ABSOLUTE_POS] =
            quantize_symmetric_int8_packed::<F>(values, scale, range_min, range_max);
    }
}

fn create_quantized_output<R: CubeRuntime>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
    scheme: QuantScheme,
) -> (CubeTensor<R>, CubeTensor<R>) {
    // Scale and offset (optional) qparams are also packed in the tensor data
    let (shapes, elem_sizes) = match &scheme {
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            ..
        } => {
            let shapes = vec![shape.dims.as_slice(), &[1]];
            let elem_sizes = vec![size_of::<i8>(), size_of::<f32>()];
            (shapes, elem_sizes)
        }
    };

    let mut tensors = client.empty_tensors(shapes, elem_sizes).into_iter();
    let (handle, strides) = tensors.next().unwrap();
    let (scale, scale_stride) = tensors.next().unwrap();

    let data = CubeTensor::new(
        client.clone(),
        handle,
        shape,
        device.clone(),
        strides,
        burn_tensor::DType::QFloat(scheme),
    );
    let scale = CubeTensor::new(
        client,
        scale,
        Shape::new([1]),
        device,
        scale_stride,
        burn_tensor::DType::F32,
    );
    (data, scale)
}

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R, F>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
{
    let (output, out_scale) = create_quantized_output(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
        *scheme,
    );

    if i8::is_supported(&tensor.client) {
        quantize_unpacked::<R, F>(tensor, scheme, scale, output, out_scale)
    } else {
        quantize_packed::<R, F>(tensor, scheme, scale, output, out_scale)
    }
}

fn quantize_unpacked<R: CubeRuntime, F: FloatElement>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
    output: CubeTensor<R>,
    out_scale: CubeTensor<R>,
) -> CubeTensor<R> {
    let client = tensor.client.clone();
    // Output tensor contains 4x less elements (four int8 values packed in a single u32)
    let num_elems = tensor.shape.num_elements();

    let out_layout = strided_layout(&output);

    // Force vectorization to process 4 quantized values packed for 1 output value
    let line_size = max_line_size(&tensor);
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    println!("output handle: {:?}", output.handle);
    println!("out_scale handle: {:?}", out_scale.handle);

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            ..
        } => {
            unsafe {
                quantize_per_tensor_symmetric_int8_kernel::launch_unchecked::<F, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    tensor.as_tensor_arg::<F>(line_size),
                    scale.as_array_arg::<f32>(1),
                    ScalarArg::new(F::from_int(-i8::MAX as i64)),
                    ScalarArg::new(F::from_int(i8::MAX as i64)),
                    output.as_tensor_arg::<i8>(line_size),
                    out_scale.as_array_arg::<f32>(1),
                    out_layout,
                    Some(tensor.shape.num_dims() as u32),
                )
            };
        }
    }

    output
}

fn quantize_packed<R: CubeRuntime, F: FloatElement>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
    output: CubeTensor<R>,
    out_scale: CubeTensor<R>,
) -> CubeTensor<R> {
    let tensor = into_contiguous(tensor);
    let client = tensor.client.clone();
    // Output tensor contains 4x less elements (four int8 values packed in a single u32)
    let num_elems = tensor.shape.num_elements();

    // Force vectorization to process 4 quantized values packed for 1 output value
    let line_size: u8 = 1;
    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems.div_ceil(line_size as usize), cube_dim);

    println!("scale: {}", into_data_sync::<R, f32>(scale.clone()));

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            ..
        } => {
            unsafe {
                quantize_per_tensor_symmetric_int8_packed_kernel::launch_unchecked::<F, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    tensor.as_tensor_arg::<F>(line_size),
                    scale.as_array_arg::<f32>(1),
                    ScalarArg::new(F::from_int(-i8::MAX as i64)),
                    ScalarArg::new(F::from_int(i8::MAX as i64)),
                    output.as_array_arg::<u32>(1),
                    out_scale.as_array_arg::<f32>(1),
                )
            };
        }
    }

    println!("out_scale: {}", into_data_sync::<R, f32>(out_scale.clone()));

    output
}
