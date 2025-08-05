use crate::{
    CubeRuntime, FloatElement,
    kernel::into_contiguous,
    ops::{empty_qtensor, max_line_size},
};
use crate::{kernel::utils::strided_layout, tensor::CubeTensor};
use burn_tensor::quantization::{
    QuantFloatPrecision, QuantInputType, QuantLevel, QuantMode, QuantScheme, QuantStoreType,
};
use burn_tensor::{bf16, f16};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::std::tensor::{StridedLayout, index_offset_contiguous};

#[cube]
fn quantize_symmetric<F: Float, FS: Float>(
    value: Line<F>,
    scale: FS,
    range_min: F,
    range_max: F,
) -> Line<F> {
    // x_q = clamp(round(x / scale), a, b)
    Line::clamp(
        Line::round(value / Line::cast_from(scale)),
        Line::new(range_min),
        Line::new(range_max),
    )
}

#[cube]
fn quantize_symmetric_i<F: Float, FS: Float, I: Int>(
    value: Line<F>,
    scale: FS,
    range_min: F,
    range_max: F,
) -> Line<I> {
    Line::cast_from(quantize_symmetric::<F, FS>(
        value, scale, range_min, range_max,
    ))
}

#[cube]
fn quantize_packed_value<F: Float, FS: Float, QS: Int>(
    value: Line<F>,
    scale: FS,
    range_min: F,
    range_max: F,
    #[comptime] scheme: QuantScheme,
) -> QS {
    let value = quantize_symmetric::<F, FS>(value, scale, range_min, range_max);
    pack_q::<F, QS>(value, scheme.q_type)
}

/// Pack a line of quantized floating-point values into a single integer (the stored quantization type),
/// according to the specified quantization input type.
#[allow(clippy::explicit_counter_loop)]
#[cube]
fn pack_q<F: Float, QS: Int>(value: Line<F>, #[comptime] quant: QuantInputType) -> QS {
    let size_quant = comptime!(quant.size_bits() as u32);

    let size_store = comptime!(QS::size_bits().unwrap() as u32);
    let num_quants = comptime!(size_store / size_quant);

    let mask = i32::cast_from(comptime!((1 << size_quant) - 1));
    let mut position = comptime!(0);
    let mut packed = QS::cast_from(0);

    // Shift and combine into QS (using i32 for sign extension)
    #[unroll]
    for _ in 0..num_quants {
        let offset = QS::cast_from(comptime!(position * size_quant));
        let shifted = QS::cast_from(i32::cast_from(value[position]) & mask) << offset;
        packed |= shifted;
        comptime!(position += 1);
    }

    packed
}

#[cube]
fn write_scale_per_tensor<F: Float, FS: Float>(
    in_pos: u32,
    scale: &Array<F>,
    out_scale: &mut Array<FS>,
) -> FS {
    let scale = FS::cast_from(scale[0]);

    // Write the scale into the output buffer
    if in_pos == 0 {
        out_scale[in_pos] = scale;
    }

    scale
}

#[cube]
fn write_scale_per_block<F: Float, FS: Float>(
    in_pos: u32,
    scale: &Array<F>,
    out_scale: &mut Array<FS>,
    #[comptime] block_size: u32,
) -> FS {
    let scale_pos = in_pos / block_size;
    let scale = FS::cast_from(scale[scale_pos]);

    // Write the scale into the output buffer
    if in_pos % block_size == 0 {
        out_scale[scale_pos] = scale;
    }

    scale
}

#[cube(launch_unchecked)]
fn quantize_symmetric_int8_native_kernel<F: Float, FS: Float>(
    input: &Tensor<Line<F>>,
    scale: &Array<F>,
    range_min: F,
    range_max: F,
    output: &mut Tensor<Line<i8>>,
    out_scale: &mut Array<FS>,
    out_layout: StridedLayout,
    #[comptime] rank: Option<u32>,
    #[comptime] scheme: QuantScheme,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let in_pos = index_offset_contiguous(input, ABSOLUTE_POS, rank);
    let out_pos = out_layout.index(output, ABSOLUTE_POS);

    let scale = match comptime!(scheme) {
        QuantScheme {
            level: QuantLevel::Block(block_size),
            ..
        } => write_scale_per_block(
            in_pos * input.line_size(),
            scale,
            out_scale,
            comptime!(block_size as u32),
        ),
        QuantScheme {
            level: QuantLevel::Tensor,
            ..
        } => write_scale_per_tensor(ABSOLUTE_POS, scale, out_scale),
    };

    output[out_pos] = quantize_symmetric_i::<F, FS, i8>(input[in_pos], scale, range_min, range_max);
}

#[cube(launch_unchecked)]
fn quantize_symmetric_int8_packed_kernel<F: Float, FS: Float>(
    input: &Tensor<Line<F>>,
    scale: &Array<F>,
    range_min: F,
    range_max: F,
    output: &mut Array<u32>,
    out_scale: &mut Array<FS>,
    #[comptime] scheme: QuantScheme,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let num_quants = comptime!((scheme.size_bits_stored() / scheme.q_type.size_bits()) as u32);
    let packed_pos = ABSOLUTE_POS * num_quants;

    let scale = match comptime!(scheme) {
        QuantScheme {
            level: QuantLevel::Block(block_size),
            ..
        } => write_scale_per_block(packed_pos, scale, out_scale, comptime!(block_size as u32)),
        QuantScheme {
            level: QuantLevel::Tensor,
            ..
        } => write_scale_per_tensor(ABSOLUTE_POS, scale, out_scale),
    };

    if comptime!(input.line_size() == num_quants) {
        output[ABSOLUTE_POS] = quantize_packed_value::<F, FS, u32>(
            input[ABSOLUTE_POS],
            scale,
            range_min,
            range_max,
            scheme,
        );
    } else {
        // Input line size = 1
        let mut values = Line::<F>::empty(num_quants);
        #[unroll]
        for i in 0..num_quants {
            values[i] = input[packed_pos + i][0];
        }
        output[ABSOLUTE_POS] =
            quantize_packed_value::<F, FS, u32>(values, scale, range_min, range_max, scheme);
    }
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
    let output = empty_qtensor(tensor.shape.clone(), *scheme, &tensor.device);

    match scheme {
        QuantScheme {
            q_type: QuantInputType::QInt8,
            q_store_type: QuantStoreType::U32,
            ..
        } => match scheme.q_params_precision {
            QuantFloatPrecision::F32 => quantize_packed::<R, F, f32>(tensor, scheme, scale, output),
            QuantFloatPrecision::F16 => quantize_packed::<R, F, f16>(tensor, scheme, scale, output),
            QuantFloatPrecision::BF16 => {
                quantize_packed::<R, F, bf16>(tensor, scheme, scale, output)
            }
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
                QuantFloatPrecision::F32 => {
                    quantize_native::<R, F, f32>(tensor, scheme, scale, output)
                }
                QuantFloatPrecision::F16 => {
                    quantize_native::<R, F, f16>(tensor, scheme, scale, output)
                }
                QuantFloatPrecision::BF16 => {
                    quantize_native::<R, F, bf16>(tensor, scheme, scale, output)
                }
            }
        }
    }
}

fn quantize_native<R: CubeRuntime, F: FloatElement, FS: FloatElement>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
    output: CubeTensor<R>,
) -> CubeTensor<R> {
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let out_layout = strided_layout(&output);
    let out_scale = output.scales().unwrap();

    let line_size = max_line_size(&tensor);
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            q_store_type: QuantStoreType::Native,
            ..
        } => {
            // We could use line_size = block_size if it's in the supported line sizes.. but let's keep it simple
            super::check_block_size_compat(scheme, line_size as usize);
            unsafe {
                quantize_symmetric_int8_native_kernel::launch_unchecked::<F, FS, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    tensor.as_tensor_arg::<F>(line_size),
                    // scale is computed based on input float dtype, but stored based on qparams precision
                    scale.as_array_arg::<F>(1),
                    ScalarArg::new(F::from_int(-i8::MAX as i64)),
                    ScalarArg::new(F::from_int(i8::MAX as i64)),
                    output.as_tensor_arg::<i8>(line_size),
                    out_scale.as_array_arg::<FS>(1),
                    out_layout,
                    Some(tensor.shape.num_dims() as u32),
                    *scheme,
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

fn quantize_packed<R: CubeRuntime, F: FloatElement, FS: FloatElement>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
    output: CubeTensor<R>,
) -> CubeTensor<R> {
    let tensor = into_contiguous(tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let out_scale = output.scales().unwrap();

    // Input line size selected based on the number of packed values per storage type
    let num_quants = (scheme.size_bits_stored() / scheme.q_type.size_bits()) as u8;
    let use_packed_line_size =
        num_elems % num_quants as usize == 0 && R::supported_line_sizes().contains(&num_quants);

    let line_size = if use_packed_line_size { num_quants } else { 1 };

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems.div_ceil(line_size as usize), cube_dim);

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            q_store_type: QuantStoreType::U32,
            ..
        } => {
            super::check_block_size_compat(scheme, num_quants as usize); // 32 / 8 = 4
            unsafe {
                quantize_symmetric_int8_packed_kernel::launch_unchecked::<F, FS, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    tensor.as_tensor_arg::<F>(line_size),
                    // scale is computed based on input float dtype, but stored based on qparams precision
                    scale.as_array_arg::<F>(1),
                    ScalarArg::new(F::from_int(-i8::MAX as i64)),
                    ScalarArg::new(F::from_int(i8::MAX as i64)),
                    output.as_array_arg::<u32>(1),
                    out_scale.as_array_arg::<FS>(1),
                    *scheme,
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
