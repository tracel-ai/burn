//! Quantized tensor operations for the Flex backend.

use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use burn_backend::{
    DType, ExecutionError, FloatDType, TensorData, TensorMetadata,
    ops::{IntTensorOps, QTensorOps},
    quantization::{
        BlockLayout, BlockSize, PermuteQuantScheme, QuantCodes, QuantScheme, QuantStore,
        QuantizationParametersPrimitive, QuantizedBytes, ScaleDtype, global_scale_dtype,
        scale_to_dtype,
    },
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{Bytes, Shape, Slice, bf16, f16};

use super::float_storage_as_f32;
use crate::{Flex, FlexQTensor, FlexTensor, Layout};

/// The blocks over `shape`, which must be a whole number of blocks along every axis.
fn block_layout(shape: &Shape, block: &BlockSize) -> BlockLayout {
    let blocks = BlockLayout::new(shape, block);
    assert!(
        blocks.divides(),
        "tensor {shape:?} is not a whole number of {block:?} blocks"
    );
    blocks
}

/// The largest magnitude in each block of `values`, laid out as `blocks`.
fn block_max_abs(values: &[f32], blocks: &BlockLayout) -> Vec<f32> {
    let mut peaks = alloc::vec![0.0f32; blocks.num_blocks()];
    for (index, &x) in values.iter().enumerate() {
        let peak = &mut peaks[blocks.block_of(index)];
        *peak = peak.max(x.abs());
    }
    peaks
}

impl QTensorOps<Flex> for Flex {
    fn q_from_data(data: TensorData, _device: &Device<Flex>) -> QuantizedTensor<Flex> {
        let scheme = match data.dtype() {
            DType::QFloat(scheme) => scheme,
            _ => panic!("Expected quantized dtype, got {:?}", data.dtype()),
        };

        let shape = data.shape().clone();

        let q_bytes = QuantizedBytes {
            shape: shape.clone(),
            bytes: data.into_bytes(),
            scheme,
        };

        let (values, qparams) = q_bytes.into_vec_i8();
        let tensor_data = TensorData::new(values, shape);
        let tensor = FlexTensor::from_data(tensor_data);

        FlexQTensor::new(tensor, scheme, qparams.block, qparams.global)
    }

    fn quantize_dynamic(tensor: FloatTensor<Flex>, scheme: &QuantScheme) -> QuantizedTensor<Flex> {
        let shape = tensor.shape();
        let tensor = tensor.to_contiguous();
        let float_data = float_storage_as_f32(&tensor);
        let (min, max) = scheme.value.range();
        let range = max - min;

        let (quantized, scales, global) = match (scheme.block_size(), global_scale_dtype(scheme)) {
            (Some(block), Some(global_dtype)) => {
                let blocks = block_layout(&shape, &block);
                let raw: Vec<f32> = block_max_abs(&float_data, &blocks)
                    .into_iter()
                    .map(|alpha| 2.0 * alpha / range)
                    .collect();
                let peak = raw.iter().copied().fold(0.0f32, f32::max);

                let global = validated_scale(
                    peak / scheme.scale_dtype().max_representable(),
                    global_dtype,
                );

                let scales: Vec<f32> = raw
                    .iter()
                    .map(|&raw| validated_scale(raw / global, scheme.scale_dtype()))
                    .collect();
                let quantized = scheme
                    .value
                    .encode_all(&float_data, |index| global * scales[blocks.block_of(index)]);

                (quantized, scales, Some(global))
            }
            (None, _) => {
                let scale = validated_scale(
                    block_max_abs_scale(&float_data, range),
                    scheme.scale_dtype(),
                );
                let quantized = scheme.value.encode_all(&float_data, move |_| scale);

                (quantized, alloc::vec![scale], None)
            }
            (Some(block_size), None) => {
                let blocks = block_layout(&shape, &block_size);
                let scales: Vec<f32> = block_max_abs(&float_data, &blocks)
                    .into_iter()
                    .map(|alpha| validated_scale(2.0 * alpha / range, scheme.scale_dtype()))
                    .collect();
                let quantized = scheme
                    .value
                    .encode_all(&float_data, |index| scales[blocks.block_of(index)]);

                (quantized, scales, None)
            }
        };

        let bytes = Bytes::from_elems(quantized);
        let layout = Layout::contiguous(shape);
        let qt = FlexTensor::new(bytes, layout, DType::I8);

        FlexQTensor::new(qt, *scheme, scales, global)
    }

    fn quantize(
        tensor: FloatTensor<Flex>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Flex>,
    ) -> QuantizedTensor<Flex> {
        let shape = tensor.shape();
        let tensor = tensor.to_contiguous();
        let float_data = float_storage_as_f32(&tensor);

        // Extract and validate scales from the qparams tensor. The scales tensor
        // shares its dtype with the float element type, which can be any of
        // f32/f64/f16/bf16, so we normalise via float_storage_as_f32 instead of
        // assuming f32 storage.
        let scales_tensor = qparams.scales.to_contiguous();
        let scales_data = float_storage_as_f32(&scales_tensor);
        let scales: Vec<f32> = scales_data
            .iter()
            .copied()
            .map(|s| validated_scale(s, scheme.scale_dtype()))
            .collect();

        let global = qparams.global.map(|global| {
            let dtype = global_scale_dtype(scheme)
                .expect("a per-tensor scale should come with a two-level scheme");
            let global = global.to_contiguous();
            validated_scale(float_storage_as_f32(&global)[0], dtype)
        });

        let quantized = match scheme.block_size() {
            None => {
                let scale = scales[0];
                scheme.value.encode_all(&float_data, move |_| scale)
            }
            Some(block_size) => {
                let blocks = block_layout(&shape, &block_size);
                let multiplier = global.unwrap_or(1.0);
                scheme.value.encode_all(&float_data, |index| {
                    multiplier * scales[blocks.block_of(index)]
                })
            }
        };

        let bytes = Bytes::from_elems(quantized);
        let layout = Layout::contiguous(shape);
        let qt = FlexTensor::new(bytes, layout, DType::I8);

        FlexQTensor::new(qt, *scheme, scales, global)
    }

    fn dequantize(tensor: QuantizedTensor<Flex>, dtype: FloatDType) -> FloatTensor<Flex> {
        let shape = tensor.tensor.shape();
        let qt = tensor.tensor.to_contiguous();
        let q_data: &[i8] = qt.storage();

        let dequantized = match tensor.scheme.block_size() {
            None => {
                let scale = tensor.scales[0];
                tensor.scheme.value.decode_all(q_data, move |_| scale)
            }
            Some(block_size) => {
                let blocks = BlockLayout::new(&shape, &block_size);
                let multiplier = tensor.global.unwrap_or(1.0);
                tensor.scheme.value.decode_all(q_data, |index| {
                    multiplier * tensor.scales[blocks.block_of(index)]
                })
            }
        };

        let layout = Layout::contiguous(shape);
        match dtype {
            FloatDType::F32 | FloatDType::Flex32 => {
                FlexTensor::new(Bytes::from_elems(dequantized), layout, DType::F32)
            }
            FloatDType::F64 => {
                let data: Vec<f64> = dequantized.iter().map(|&v| v as f64).collect();
                FlexTensor::new(Bytes::from_elems(data), layout, DType::F64)
            }
            FloatDType::F16 => {
                let data: Vec<f16> = dequantized.iter().map(|&v| f16::from_f32(v)).collect();
                FlexTensor::new(Bytes::from_elems(data), layout, DType::F16)
            }
            FloatDType::BF16 => {
                let data: Vec<bf16> = dequantized.iter().map(|&v| bf16::from_f32(v)).collect();
                FlexTensor::new(Bytes::from_elems(data), layout, DType::BF16)
            }
        }
    }

    fn q_to_device(tensor: QuantizedTensor<Flex>, _device: &Device<Flex>) -> QuantizedTensor<Flex> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Flex>, shape: Shape) -> QuantizedTensor<Flex> {
        // Flex holds codes unpacked, so a reshape that drops the packed axis can pack innermost.
        let rank = shape.num_dims().max(1);
        let store = match tensor.scheme.store {
            QuantStore::PackedU32(packed_dim) if packed_dim >= rank => QuantStore::PackedU32(0),
            QuantStore::PackedNative(packed_dim) if packed_dim >= rank => {
                QuantStore::PackedNative(0)
            }
            store => store,
        };
        let scheme = tensor.scheme.with_store(store);
        block_safe_layout_op(tensor, scheme, |t| t.reshape(shape))
    }

    async fn q_into_data(tensor: QuantizedTensor<Flex>) -> Result<TensorData, ExecutionError> {
        let shape = tensor.tensor.shape();
        let scheme = tensor.scheme;
        let qt = tensor.tensor.to_contiguous();
        let values: Vec<i8> = qt.storage::<i8>().to_vec();

        Ok(TensorData::quantized(
            values,
            shape.to_vec(),
            scheme,
            &tensor.scales,
            tensor.global,
        ))
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Flex>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Flex> {
        let scheme = tensor
            .scheme
            .swapped(tensor.tensor.shape().num_dims(), dim1, dim2);
        block_safe_layout_op(tensor, scheme, |t| t.transpose(dim1, dim2))
    }

    fn q_permute(tensor: QuantizedTensor<Flex>, axes: &[usize]) -> QuantizedTensor<Flex> {
        let scheme = tensor.scheme.permuted(axes);
        block_safe_layout_op(tensor, scheme, |t| t.permute(axes))
    }

    fn q_flip(tensor: QuantizedTensor<Flex>, axes: &[usize]) -> QuantizedTensor<Flex> {
        let scheme = tensor.scheme;
        block_safe_layout_op(tensor, scheme, |t| crate::ops::flip::flip(t, axes))
    }

    fn q_expand(tensor: QuantizedTensor<Flex>, shape: Shape) -> QuantizedTensor<Flex> {
        let scheme = tensor.scheme;
        block_safe_layout_op(tensor, scheme, |t| crate::ops::expand::expand(t, shape))
    }

    fn q_select(
        tensor: QuantizedTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
    ) -> QuantizedTensor<Flex> {
        match tensor.scheme.block_size() {
            None => FlexQTensor::new(
                crate::ops::gather_scatter::select::<i8>(tensor.tensor, dim, indices),
                tensor.scheme,
                tensor.scales,
                tensor.global,
            ),
            Some(_) => {
                let scheme = tensor.scheme;
                let float_tensor = Flex::dequantize(tensor, FloatDType::F32);
                let result = crate::ops::gather_scatter::select::<f32>(float_tensor, dim, indices);
                Flex::quantize_dynamic(result, &scheme)
            }
        }
    }

    fn q_slice(tensor: QuantizedTensor<Flex>, slices: &[Slice]) -> QuantizedTensor<Flex> {
        let scheme = tensor.scheme;
        block_safe_layout_op(tensor, scheme, |t| crate::ops::slice::slice(t, slices))
    }

    fn q_argmax(
        tensor: QuantizedTensor<Flex>,
        dim: usize,
        out_dtype: burn_std::IntDType,
    ) -> IntTensor<Flex> {
        let result = if codes_order_as_values(&tensor) {
            crate::ops::reduce::argmax(tensor.tensor, dim)
        } else {
            crate::ops::reduce::argmax(Flex::dequantize(tensor, FloatDType::F32), dim)
        };
        if result.dtype() != DType::from(out_dtype) {
            Flex::int_cast(result, out_dtype)
        } else {
            result
        }
    }

    fn q_argmin(
        tensor: QuantizedTensor<Flex>,
        dim: usize,
        out_dtype: burn_std::IntDType,
    ) -> IntTensor<Flex> {
        let result = if codes_order_as_values(&tensor) {
            crate::ops::reduce::argmin(tensor.tensor, dim)
        } else {
            crate::ops::reduce::argmin(Flex::dequantize(tensor, FloatDType::F32), dim)
        };
        if result.dtype() != DType::from(out_dtype) {
            Flex::int_cast(result, out_dtype)
        } else {
            result
        }
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Flex>,
        indices: IntTensor<Flex>,
    ) -> QuantizedTensor<Flex> {
        match tensor.scheme.block_size() {
            None => FlexQTensor::new(
                crate::ops::gather_scatter::gather::<i8>(tensor.tensor, dim, indices),
                tensor.scheme,
                tensor.scales,
                tensor.global,
            ),
            Some(_) => {
                let scheme = tensor.scheme;
                let float_tensor = Flex::dequantize(tensor, FloatDType::F32);
                let result = crate::ops::gather_scatter::gather::<f32>(float_tensor, dim, indices);
                Flex::quantize_dynamic(result, &scheme)
            }
        }
    }
}

/// Apply a layout operation to a quantized tensor, which takes `scheme`, already rewritten by the
/// caller to follow the move as every other backend does. A block-quantized tensor is dequantized,
/// moved and requantized, since its blocks would no longer line up with its scales.
fn block_safe_layout_op(
    qtensor: FlexQTensor,
    scheme: QuantScheme,
    op: impl FnOnce(FlexTensor) -> FlexTensor,
) -> FlexQTensor {
    match qtensor.scheme.block_size() {
        None => FlexQTensor::new(op(qtensor.tensor), scheme, qtensor.scales, qtensor.global),
        Some(_) => {
            let float_tensor = Flex::dequantize(qtensor, FloatDType::F32);
            let result = op(float_tensor);
            Flex::quantize_dynamic(result, &scheme)
        }
    }
}

/// Whether comparing `tensor`'s codes compares its values: one positive scale over integer codes.
fn codes_order_as_values(tensor: &FlexQTensor) -> bool {
    tensor.scheme.block_size().is_none() && tensor.scheme.value.codes_are_integers()
}

/// Unrounded; callers round separately.
fn block_max_abs_scale(block: &[f32], range: f32) -> f32 {
    let alpha = block.iter().fold(0.0f32, |alpha, &x| alpha.max(x.abs()));
    2.0 * alpha / range
}

/// Only an exactly zero scale is replaced: the dtype's subnormals carry real information for a
/// small tensor, and the replacement goes back through the dtype because `f32::MIN_POSITIVE`
/// would itself encode to zero in a narrow one.
fn validated_scale(scale: f32, dtype: ScaleDtype) -> f32 {
    let scale = scale_to_dtype(scale, dtype);
    if scale > 0.0 && scale.is_finite() {
        scale
    } else {
        scale_to_dtype(f32::MIN_POSITIVE, dtype)
    }
}

// Tests kept here exercise flex-specific behavior: quantization scheme
// roundtrips, per-block / dynamic quantization, block-quantized layout
// ops (transpose / select / flip dequantize), and f16/f64 dequantize
// dtype paths. Plain layout-preservation / select / slice / argmax /
// argmin / gather tests are covered generically in
// crates/burn-backend-tests/tests/tensor/float/quantization/ops/extended/
// so they run on every backend.
#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{TensorMetadata, quantization::QuantValue};

    fn data_of(tensor: QuantizedTensor<Flex>) -> TensorData {
        burn_std::reader::try_read_sync(Flex::q_into_data(tensor))
            .expect("flex reads synchronously")
            .unwrap()
    }

    fn packed_q4() -> QuantScheme {
        QuantScheme::default()
            .with_value(QuantValue::Q4S)
            .with_store(QuantStore::PackedU32(0))
            .per_block([8], ScaleDtype::F32)
    }

    fn ramp(shape: [usize; 2]) -> FlexTensor {
        let values = (0..shape[0] * shape[1])
            .map(|i| (i as f32 * 0.37).sin())
            .collect::<Vec<_>>();
        FlexTensor::from_data(TensorData::new(values, shape))
    }

    #[test]
    fn a_packed_tensor_keeps_its_scheme_through_its_data() {
        let scheme = packed_q4();
        let quantized = Flex::quantize_dynamic(ramp([4, 16]), &scheme);
        assert_eq!(quantized.scheme, scheme);

        let data = data_of(quantized.clone());
        assert_eq!(data.dtype(), DType::QFloat(scheme));

        let reloaded = Flex::q_from_data(data.clone(), &Default::default());
        assert_eq!(reloaded.scheme, scheme);
        assert_eq!(data_of(reloaded).as_bytes(), data.as_bytes());
    }

    #[test]
    fn swapping_the_packed_axis_moves_the_store_with_it() {
        let per_tensor = QuantScheme::default()
            .with_value(QuantValue::Q4S)
            .with_store(QuantStore::PackedU32(0));
        for scheme in [per_tensor, packed_q4()] {
            let quantized = Flex::quantize_dynamic(ramp([16, 8]), &scheme);

            let swapped = Flex::q_swap_dims(quantized, 0, 1);
            assert_eq!(swapped.scheme.store, QuantStore::PackedU32(1));

            let direct = Flex::dequantize(swapped.clone(), FloatDType::F32);
            let reloaded = Flex::q_from_data(data_of(swapped), &Default::default());
            let reloaded = Flex::dequantize(reloaded, FloatDType::F32);
            assert_eq!(
                reloaded.to_contiguous().storage::<f32>(),
                direct.to_contiguous().storage::<f32>(),
                "{scheme:?}"
            );
        }
    }

    #[test]
    fn a_reshape_that_drops_the_packed_axis_packs_innermost() {
        let q8 = QuantScheme::default().with_value(QuantValue::Q8S);
        for (scheme, flattened) in [
            (
                q8.with_store(QuantStore::PackedU32(0)),
                QuantStore::PackedU32(0),
            ),
            (
                q8.with_value(QuantValue::E2M1)
                    .with_store(QuantStore::PackedNative(0)),
                QuantStore::PackedNative(0),
            ),
        ] {
            let swapped = Flex::q_swap_dims(Flex::quantize_dynamic(ramp([4, 8]), &scheme), 0, 1);
            let expected = Flex::dequantize(swapped.clone(), FloatDType::F32);

            let reshaped = Flex::q_reshape(swapped, Shape::new([32]));

            assert_eq!(reshaped.scheme.store, flattened);
            let reloaded = Flex::q_from_data(data_of(reshaped), &Default::default());
            assert_eq!(
                Flex::dequantize(reloaded, FloatDType::F32).storage::<f32>(),
                expected.to_contiguous().storage::<f32>(),
                "{scheme:?}"
            );
        }
    }

    #[test]
    fn global_reductions_follow_a_swap_of_the_packed_axis() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::PackedU32(0));
        let swapped = Flex::q_swap_dims(Flex::quantize_dynamic(ramp([4, 8]), &scheme), 0, 1);
        let values = Flex::dequantize(swapped.clone(), FloatDType::F32)
            .to_contiguous()
            .storage::<f32>()
            .to_vec();
        let reduced = |tensor: QuantizedTensor<Flex>| {
            Flex::dequantize(tensor, FloatDType::F32).storage::<f32>()[0]
        };

        assert_eq!(
            reduced(Flex::q_max(swapped.clone())),
            values.iter().copied().fold(f32::MIN, f32::max)
        );
        assert_eq!(
            reduced(Flex::q_min(swapped.clone())),
            values.iter().copied().fold(f32::MAX, f32::min)
        );
        assert_eq!(
            reduced(Flex::q_max_abs(swapped)).abs(),
            values.iter().map(|value| value.abs()).fold(0.0, f32::max)
        );
    }

    #[test]
    fn float_formats_reconstruct_values_on_their_grid() {
        for (value, values) in [
            (
                QuantValue::E2M1,
                [0.0, 0.5, -1.5, 6.0, -3.0, 2.0, 4.0, -6.0],
            ),
            (
                QuantValue::E4M3,
                [448.0, -2.5, 0.125, 1.0, -0.0, 3.5, -448.0, 64.0],
            ),
        ] {
            let scheme = QuantScheme::default()
                .with_value(value)
                .with_store(QuantStore::Native);
            let input = FlexTensor::from_data(TensorData::new(values.to_vec(), [8]));

            let output = Flex::dequantize(Flex::quantize_dynamic(input, &scheme), FloatDType::F32);

            assert_eq!(output.to_contiguous().storage::<f32>(), values, "{value:?}");
        }
    }

    #[test]
    fn argmax_over_float_codes_follows_the_values() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .with_store(QuantStore::Native);
        let input = FlexTensor::from_data(TensorData::new(vec![-6.0f32, -1.0, 0.5, -0.5], [1, 4]));
        let quantized = Flex::quantize_dynamic(input, &scheme);

        let argmax = Flex::q_argmax(quantized.clone(), 1, burn_std::IntDType::I64);
        let argmin = Flex::q_argmin(quantized, 1, burn_std::IntDType::I64);

        assert_eq!(argmax.to_contiguous().storage::<i64>(), [2]);
        assert_eq!(argmin.to_contiguous().storage::<i64>(), [0]);
    }

    #[test]
    fn float_schemes_keep_their_layout_through_their_data() {
        let mxfp4 = QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .per_block([32], ScaleDtype::UE8M0);
        for scheme in [
            QuantScheme::default().with_value(QuantValue::E4M3),
            QuantScheme::default()
                .with_value(QuantValue::E5M2)
                .with_store(QuantStore::Native),
            mxfp4,
            mxfp4.with_store(QuantStore::PackedNative(0)),
        ] {
            let quantized = Flex::quantize_dynamic(ramp([4, 32]), &scheme);
            let data = data_of(quantized.clone());
            assert_eq!(data.dtype(), DType::QFloat(scheme));

            let reloaded = Flex::q_from_data(data.clone(), &Default::default());
            assert_eq!(
                data_of(reloaded.clone()).as_bytes(),
                data.as_bytes(),
                "{scheme:?}"
            );
            assert_eq!(
                Flex::dequantize(reloaded, FloatDType::F32)
                    .to_contiguous()
                    .storage::<f32>(),
                Flex::dequantize(quantized, FloatDType::F32)
                    .to_contiguous()
                    .storage::<f32>(),
                "{scheme:?}"
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // Create a float tensor
        let values = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [2, 3]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        // Compute scale: symmetric, so scale = 2 * max(|min|, |max|) / (b - a)
        // max_abs = 5.0, range = 127 - (-127) = 254
        // scale = 2 * 5.0 / 254 = 0.03937008
        let scale: f32 = 2.0 * 5.0 / 254.0;
        let scales_tensor = FlexTensor::from_data(TensorData::new(vec![scale], [1]));

        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
            global: None,
        };

        // Quantize
        let qtensor = Flex::quantize(tensor, &scheme, qparams);
        assert_eq!(qtensor.tensor.shape().to_vec(), vec![2, 3]);
        assert_eq!(qtensor.tensor.dtype(), DType::I8);

        // Check quantized values
        let q_vals: &[i8] = qtensor.tensor.storage();
        // 0 / 0.03937 = 0, 1 / 0.03937 = 25.4 -> 25, etc.
        assert_eq!(q_vals[0], 0);
        assert_eq!(q_vals[1], 25);
        assert_eq!(q_vals[5], 127);

        // Dequantize
        let result = Flex::dequantize(qtensor, FloatDType::F32);
        assert_eq!(result.shape().to_vec(), vec![2, 3]);
        assert_eq!(result.dtype(), DType::F32);

        let result_vals: &[f32] = result.storage();
        // Values should be approximately equal (quantization introduces small errors)
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.05, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_quantize_dequantize_negative_values() {
        let values = vec![-3.0f32, -1.5, 0.0, 1.5, 3.0];
        let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [5]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let scale: f32 = 2.0 * 3.0 / 254.0;
        let scales_tensor = FlexTensor::from_data(TensorData::new(vec![scale], [1]));

        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
            global: None,
        };

        let qtensor = Flex::quantize(tensor, &scheme, qparams);
        let result = Flex::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();

        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.05, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_q_from_data_into_data_roundtrip() {
        // Create quantized TensorData the standard way
        let values = vec![0i8, 25, 51, 76, 102, 127];
        let scale = 0.03937008f32;
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let data = TensorData::quantized(values.clone(), [2, 3], scheme, &[scale], None);

        // Load into FlexQTensor
        let qtensor = Flex::q_from_data(data, &Default::default());
        assert_eq!(qtensor.tensor.shape().to_vec(), vec![2, 3]);
        assert_eq!(qtensor.scales, vec![scale]);

        // Dequantize and check values
        let float_tensor = Flex::dequantize(qtensor, FloatDType::F32);
        let result: &[f32] = float_tensor.storage();
        assert!((result[0]).abs() < 0.01); // 0 * scale ~ 0
        assert!((result[5] - 5.0).abs() < 0.05); // 127 * scale ~ 5.0
    }

    #[test]
    fn test_quantize_zero_tensor() {
        let values = vec![0.0f32; 4];
        let tensor = FlexTensor::from_data(TensorData::new(values, [4]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        // Scale of 0 should be handled gracefully
        let scales_tensor = FlexTensor::from_data(TensorData::new(vec![0.0f32], [1]));
        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
            global: None,
        };

        let qtensor = Flex::quantize(tensor, &scheme, qparams);
        let q_vals: &[i8] = qtensor.tensor.storage();
        assert_eq!(q_vals, &[0, 0, 0, 0]);
    }

    /// The scale a narrow dtype can represent is coarser than the exact one, so the stored scale
    /// must reflect the dtype rather than staying at full `f32` precision. Before scales were
    /// rounded, every scale dtype produced byte-identical results here.
    #[test]
    fn test_quantize_dynamic_honors_scale_dtype_precision() {
        // 4.5 / 127 is not representable in e4m3, so rounding is observable.
        let values = vec![-3.0f32, -1.5, 0.0, 1.5, 3.0, 4.5];
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let quantize_with = |dtype| {
            let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [2, 3]));
            Flex::quantize_dynamic(tensor, &scheme.per_tensor(dtype)).scales[0]
        };

        let exact = quantize_with(ScaleDtype::F32);
        let coarse = quantize_with(ScaleDtype::UE4M3);

        assert_ne!(
            exact, coarse,
            "UE4M3 scale should differ from the exact f32 scale"
        );
        assert_eq!(
            coarse,
            scale_to_dtype(exact, ScaleDtype::UE4M3),
            "stored scale should be the exact scale rounded to the scale dtype"
        );
    }

    #[test]
    fn test_quantize_dynamic_roundtrip() {
        let values = vec![-3.0f32, -1.5, 0.0, 1.5, 3.0, 4.5];
        let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [2, 3]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let qtensor = Flex::quantize_dynamic(tensor, &scheme);
        assert_eq!(qtensor.tensor.shape().to_vec(), vec![2, 3]);
        assert_eq!(qtensor.scales.len(), 1);

        // Scale should be 2 * 4.5 / 254
        let expected_scale: f32 = 2.0 * 4.5 / 254.0;
        assert!(
            (qtensor.scales[0] - expected_scale).abs() < 1e-6,
            "scale={}, expected={}",
            qtensor.scales[0],
            expected_scale
        );

        let result = Flex::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.1, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_per_block_quantize_dequantize() {
        use burn_std::quantization::BlockSize;

        let values = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [8]));

        let block_size = BlockSize::new([4]);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .per_block(block_size.as_slice(), ScaleDtype::F32)
            .with_store(QuantStore::Native);

        // Block 1: [0, 1, 2, 3] -> max_abs=3, scale = 6/254
        // Block 2: [4, 5, 6, 7] -> max_abs=7, scale = 14/254
        let scale_1: f32 = 2.0 * 3.0 / 254.0;
        let scale_2: f32 = 2.0 * 7.0 / 254.0;
        let scales_tensor = FlexTensor::from_data(TensorData::new(vec![scale_1, scale_2], [2]));

        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
            global: None,
        };

        let qtensor = Flex::quantize(tensor, &scheme, qparams);
        assert_eq!(qtensor.scales.len(), 2);

        let result = Flex::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();

        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.1, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_quantize_dynamic_block() {
        use burn_std::quantization::BlockSize;

        let values = vec![-2.0f32, -1.0, 0.0, 1.0, 4.0, 5.0, 6.0, 7.0];
        let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [8]));

        let block_size = BlockSize::new([4]);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .per_block(block_size.as_slice(), ScaleDtype::F32)
            .with_store(QuantStore::Native);

        let qtensor = Flex::quantize_dynamic(tensor, &scheme);
        assert_eq!(qtensor.scales.len(), 2);

        // Block 1: [-2, -1, 0, 1] -> alpha=2, scale = 4/254
        // Block 2: [4, 5, 6, 7] -> alpha=7, scale = 14/254
        let expected_scale_1: f32 = 2.0 * 2.0 / 254.0;
        let expected_scale_2: f32 = 2.0 * 7.0 / 254.0;
        assert!((qtensor.scales[0] - expected_scale_1).abs() < 1e-6);
        assert!((qtensor.scales[1] - expected_scale_2).abs() < 1e-6);

        let result = Flex::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.1, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_quantize_dynamic_q8f() {
        // Q8F uses asymmetric range [-128, 127]
        let values = vec![-5.0f32, -2.5, 0.0, 2.5, 5.0, 7.5];
        let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [6]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8F)
            .with_store(QuantStore::Native);

        let qtensor = Flex::quantize_dynamic(tensor, &scheme);

        // Q8F range: [-128, 127], so range = 255
        // alpha = 7.5, scale = 2 * 7.5 / 255
        let expected_scale: f32 = 2.0 * 7.5 / 255.0;
        assert!(
            (qtensor.scales[0] - expected_scale).abs() < 1e-6,
            "scale={}, expected={}",
            qtensor.scales[0],
            expected_scale
        );

        let result = Flex::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.1, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_block_quantized_transpose_dequantize() {
        use burn_std::quantization::BlockSize;

        // 2x4 tensor, 2 blocks of 4
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = FlexTensor::from_data(TensorData::new(values, [2, 4]));

        let block_size = BlockSize::new([4]);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .per_block(block_size.as_slice(), ScaleDtype::F32)
            .with_store(QuantStore::Native);

        let qtensor = Flex::quantize_dynamic(tensor, &scheme);

        // Transpose to [4, 2], then dequantize
        let transposed = Flex::q_swap_dims(qtensor, 0, 1);
        assert_eq!(transposed.tensor.shape().to_vec(), vec![4, 2]);

        let result = Flex::dequantize(transposed, FloatDType::F32);
        let result_vals: &[f32] = result.storage();

        // Original [[1,2,3,4],[5,6,7,8]] transposed to [[1,5],[2,6],[3,7],[4,8]]
        let expected = [1.0f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
        for (exp, deq) in expected.iter().zip(result_vals.iter()) {
            assert!(
                (exp - deq).abs() < 0.15,
                "expected={exp}, dequantized={deq}"
            );
        }
    }

    #[test]
    fn test_block_quantized_select() {
        use burn_std::quantization::BlockSize;

        // 2x4 tensor, 2 blocks of 4
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let tensor = FlexTensor::from_data(TensorData::new(values, [2, 4]));

        let block_size = BlockSize::new([4]);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .per_block(block_size.as_slice(), ScaleDtype::F32)
            .with_store(QuantStore::Native);

        let qtensor = Flex::quantize_dynamic(tensor, &scheme);

        // Select row 1 -> [10, 20, 30, 40]
        let indices = FlexTensor::from_data(TensorData::new(vec![1i64], [1]));
        let selected = Flex::q_select(qtensor, 0, indices);
        assert_eq!(selected.tensor.shape().to_vec(), vec![1, 4]);

        let result = Flex::dequantize(selected, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        let expected = [10.0f32, 20.0, 30.0, 40.0];
        for (exp, deq) in expected.iter().zip(result_vals.iter()) {
            assert!((exp - deq).abs() < 0.5, "expected={exp}, dequantized={deq}");
        }
    }

    #[test]
    fn test_block_quantized_flip_dequantize() {
        use burn_std::quantization::BlockSize;

        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = FlexTensor::from_data(TensorData::new(values, [2, 4]));

        let block_size = BlockSize::new([4]);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .per_block(block_size.as_slice(), ScaleDtype::F32)
            .with_store(QuantStore::Native);

        let qtensor = Flex::quantize_dynamic(tensor, &scheme);

        // Flip along axis 0: [[5,6,7,8],[1,2,3,4]]
        let flipped = Flex::q_flip(qtensor, &[0]);
        assert_eq!(flipped.tensor.shape().to_vec(), vec![2, 4]);

        let result = Flex::dequantize(flipped, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        let expected = [5.0f32, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0];
        for (exp, deq) in expected.iter().zip(result_vals.iter()) {
            assert!(
                (exp - deq).abs() < 0.15,
                "expected={exp}, dequantized={deq}"
            );
        }
    }

    #[test]
    fn test_quantize_dynamic_f64_tensor() {
        use burn_backend::quantization::QuantValue;

        let values = vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = FlexTensor::new(
            Bytes::from_elems(values),
            Layout::contiguous([6].into()),
            DType::F64,
        );
        assert_eq!(tensor.dtype(), DType::F64);

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let qtensor = Flex::quantize_dynamic(tensor, &scheme);
        assert_eq!(qtensor.tensor.dtype(), DType::I8);

        // Dequantize and verify round-trip accuracy
        let result = Flex::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        let expected = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        for (exp, deq) in expected.iter().zip(result_vals.iter()) {
            assert!(
                (exp - deq).abs() < 0.15,
                "expected={exp}, dequantized={deq}"
            );
        }
    }

    #[test]
    fn test_dequantize_f64() {
        let values = vec![0.0f32, 1.0, 2.0, 3.0];
        let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [4]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let qtensor = Flex::quantize_dynamic(tensor, &scheme);
        let result = Flex::dequantize(qtensor, FloatDType::F64);
        assert_eq!(result.dtype(), DType::F64);
        let result_vals: &[f64] = result.storage();
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!(
                (*orig as f64 - deq).abs() < 0.05,
                "orig={orig}, dequantized={deq}"
            );
        }
    }

    #[test]
    fn test_dequantize_f16() {
        let values = vec![0.0f32, 1.0, 2.0, 3.0];
        let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [4]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let qtensor = Flex::quantize_dynamic(tensor, &scheme);
        let result = Flex::dequantize(qtensor, FloatDType::F16);
        assert_eq!(result.dtype(), DType::F16);
        let result_vals: &[f16] = result.storage();
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!(
                (*orig - f32::from(*deq)).abs() < 0.05,
                "orig={orig}, dequantized={deq}"
            );
        }
    }
}
