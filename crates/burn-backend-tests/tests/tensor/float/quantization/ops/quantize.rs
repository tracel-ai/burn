use super::*;
use alloc::{vec, vec::Vec};
use burn_tensor::Tolerance;
use burn_tensor::quantization::{
    DecodedScales, QuantScheme, QuantStore, QuantValue, QuantizationParameters, QuantizedBytes,
    ScaleDtype,
};
use burn_tensor::{DType, Element, TensorData};

fn get_q_params(data: TensorData) -> DecodedScales {
    let scheme = if let DType::QFloat(scheme) = data.dtype {
        scheme
    } else {
        unreachable!()
    };
    let q_bytes = QuantizedBytes {
        shape: data.shape.clone(),
        bytes: data.into_bytes(),
        scheme,
    };
    q_bytes.into_vec_i8().1
}

#[test]
fn should_support_quantize_symmetric_int8() {
    // Strict equality was based on full precision
    if !matches!(FloatElem::dtype(), DType::F32) {
        return;
    }
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([-1.8, -1.0, 0.0, 0.5], &device);
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    let qparams = QuantizationParameters {
        scales: TestTensor::from_data([0.014_173_228], &device),
        global: None,
    };

    let x_q = tensor.clone().quantize(&scheme, qparams);

    let x_q_data = x_q.to_data();
    let expected = TensorData::quantized(
        vec![-127i8, -71, 0, 35],
        [4],
        scheme.with_store(QuantStore::Native),
        &[0.014_173_228], // scale,
        None,
    );

    // Values equality
    x_q_data.assert_eq(&expected, false);

    // Quantization parameters check
    let qparams = get_q_params(x_q_data);
    let expected = get_q_params(expected);
    assert_eq!(qparams.block.len(), 1);
    // TODO: check scales
    assert_eq!(qparams, expected);

    // Dequantize
    let x = x_q.dequantize();

    x.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), Tolerance::rel_abs(1e-1, 1e-2));
}

#[test]
fn should_support_quantize_dynamic_int8() {
    let device = Default::default();
    // NOTE: we use fully representable values since different backend implementations could differ slightly
    // due to rounding discrepancies
    let tensor = TestTensor::<1>::from_data([5., 0., 4., -12.7], &device);
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    let x_q = tensor.quantize_dynamic(&scheme);

    let expected = TensorData::quantized(
        vec![50i8, 0, 40, -127],
        [4],
        scheme.with_store(QuantStore::Native),
        &[0.1], // scale,
        None,
    );

    x_q.into_data().assert_eq(&expected, false);
}

#[test]
fn should_quantize_dequantize_symmetric_single_with_transform() {
    let device = Default::default();
    let input = TestTensorInt::<1>::arange(0..32, &device).float();
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    let quant = input.quantize_dynamic(&scheme);
    let result = quant * 10;

    let data = result.into_data();
    let expected = [
        0.0, 9.76378, 19.52756, 29.29134, 39.05512, 48.818897, 61.02362, 70.7874, 80.551186,
        90.31496, 100.07874, 109.84252, 119.60631, 129.37009, 139.13387, 148.89764, 161.10237,
        170.86615, 180.62991, 190.39369, 200.15749, 209.92126, 219.68504, 229.44882, 239.21262,
        248.97638, 261.1811, 270.9449, 280.70865, 290.47244, 300.23624, 310.0,
    ];
    data.assert_approx_eq::<FloatElem>(&TensorData::from(expected), Tolerance::permissive());
}

#[test]
fn should_quantize_dequantize_symmetric_arange_16x16() {
    let device = Default::default();

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &device)
        .float()
        .div_scalar(256.)
        .reshape([16, 16]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    let output = input.clone().quantize_dynamic(&scheme);
    let output = output.dequantize();

    output.into_data().assert_approx_eq::<FloatElem>(
        &input.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

#[test]
fn should_quantize_dequantize_symmetric_per_block_arange_16x16() {
    let device = Default::default();

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &device)
        .float()
        .div_scalar(256.)
        .reshape([16, 16]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .per_block([2, 16], ScaleDtype::F32);

    let output = input.clone().quantize_dynamic(&scheme);
    let output = output.dequantize();

    output.into_data().assert_approx_eq::<FloatElem>(
        &input.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

/// A block that does not span the trailing dimension is a rectangle, not a run of the flat
/// storage. Each `[2, 4]` block here holds one magnitude, so its scale is that magnitude over
/// 127; chunking the flat storage in eights would instead pair the two blocks of a row and give
/// both the larger scale. Four wide, so a packed store still keeps one word inside one block.
#[test]
fn should_quantize_blocks_that_do_not_span_the_trailing_dim() {
    let device = Default::default();

    let input = TestTensor::<2>::from_data(
        [
            [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0],
            [100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0],
        ],
        &device,
    );

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .per_block([2, 4], ScaleDtype::F32);

    let quantized = input.clone().quantize_dynamic(&scheme);

    let scales = get_q_params(quantized.to_data()).block;
    let expected = [1.0f32, 10.0, 100.0, 1000.0].map(|magnitude| magnitude / 127.0);
    TensorData::new(scales, [2, 2]).assert_approx_eq::<f32>(
        &TensorData::new(expected.to_vec(), [2, 2]),
        Tolerance::relative(1e-3),
    );

    // Every block is uniform, so a right grid reconstructs it exactly and a wrong one is off by
    // the ratio between the two magnitudes it merged.
    quantized
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&input.into_data(), Tolerance::relative(1e-2));
}

/// Bit equality rather than a tolerance: both paths reconstruct from the same stored scales, so
/// any difference means a level was rounded, folded or sliced differently on one of them.
///
/// WGSL has no e4m3, so a ue4m3 scale reconstructs as NaN there. The f16 sibling below covers the
/// byte layout on those backends.
#[cfg(not(feature = "wgpu"))]
#[test]
fn should_round_trip_two_level_through_bytes() {
    let device = Default::default();

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &device)
        .float()
        .div_scalar(256.)
        .reshape([16, 16]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .per_block([2, 16], ScaleDtype::UE4M3)
        .per_tensor(ScaleDtype::F32);

    let quantized = input.quantize_dynamic(&scheme);
    let direct = quantized.clone().dequantize().into_data();

    let reloaded = TestTensor::<2>::from_data(quantized.into_data(), &device);
    let round_tripped = reloaded.dequantize().into_data();

    round_tripped.assert_eq(&direct, true);
}

/// The per-tensor scale exists so that accuracy stops depending on how large the weights are.
///
/// WGSL has no e4m3, so a ue4m3 scale reconstructs as NaN there.
#[cfg(not(feature = "wgpu"))]
#[test]
fn two_level_error_should_not_track_weight_magnitude() {
    use burn_tensor::ElementConversion;

    let device = Default::default();

    let big: TestTensor<2> = TestTensorInt::arange(-128..128, &device)
        .float()
        .div_scalar(128.)
        .reshape([16, 16]);
    let small = big.clone().mul_scalar(0.02);

    let base = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    let two_level = base
        .per_block([2, 16], ScaleDtype::UE4M3)
        .per_tensor(ScaleDtype::F32);
    let one_level = base.per_block([2, 16], ScaleDtype::UE4M3);

    let error = |tensor: TestTensor<2>, scheme: &QuantScheme| -> f32 {
        let reference = tensor.clone();
        let dequantized = tensor.quantize_dynamic(scheme).dequantize();
        let diff: FloatElem = (reference.clone() - dequantized).abs().sum().into_scalar();
        let total: FloatElem = reference.abs().sum().into_scalar();
        diff.elem::<f32>() / total.elem::<f32>()
    };

    let big_error = error(big.clone(), &two_level);
    let small_error = error(small.clone(), &two_level);

    assert!(
        (small_error - big_error).abs() / big_error < 0.2,
        "two-level error should be the same at either magnitude, \
         got {big_error} and {small_error}"
    );

    // A one-level scheme's scales have to cover the magnitude themselves, and underflow here.
    let small_one_level = error(small, &one_level);
    assert!(
        small_one_level > small_error * 2.0,
        "one-level 8-bit scales should degrade on small weights where two-level does not, \
         got one-level {small_one_level} and two-level {small_error}"
    );
}

fn should_quantize_transposed<const D: usize>(tensor: Tensor<D>, scheme: QuantScheme) {
    let tensor_t = tensor.clone().transpose();

    let output = tensor_t.quantize_dynamic(&scheme).dequantize().transpose();

    tensor.into_data().assert_approx_eq::<FloatElem>(
        &output.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

fn should_dequantize_transposed<const D: usize>(tensor: Tensor<D>, scheme: QuantScheme) {
    let output = tensor
        .clone()
        .quantize_dynamic(&scheme)
        .transpose()
        .dequantize()
        .transpose();

    tensor.into_data().assert_approx_eq::<FloatElem>(
        &output.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

#[test]
fn should_quantize_symmetric_int8_transposed_8x32() {
    let device = Default::default();

    let tensor = TestTensorInt::arange(0..256, &device)
        .float()
        .div_scalar(256.)
        .reshape([8, 32]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    should_quantize_transposed(tensor, scheme);
}

#[test]
fn should_dequantize_symmetric_int8_transposed_8x32() {
    let device = Default::default();

    let values = (0..256)
        .map(|value| value as f32 / 256.)
        .collect::<Vec<_>>();
    let tensor = TestTensor::<2>::from_data(TensorData::new(values, [8, 32]), &device);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    should_dequantize_transposed(tensor, scheme);
}

#[test]
fn should_quantize_symmetric_int8_transposed_48x64() {
    let device = Default::default();

    let tensor = TestTensorInt::arange(0..3072, &device)
        .float()
        .div_scalar(3072.)
        .reshape([48, 64]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    should_quantize_transposed(tensor, scheme);
}

#[test]
fn should_quantize_symmetric_per_block_int8_transposed_32x64() {
    let device = Default::default();

    let tensor = TestTensorInt::arange(0..2048, &device)
        .float()
        .div_scalar(2048.)
        .reshape([32, 64]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .per_block([32], ScaleDtype::F32);
    should_quantize_transposed(tensor, scheme);
}

#[test]
fn should_dequantize_symmetric_per_block_int8_transposed_32x64() {
    let device = Default::default();

    let values = (0..2048)
        .map(|value| value as f32 / 2048.)
        .collect::<Vec<_>>();
    let tensor = TestTensor::<2>::from_data(TensorData::new(values, [32, 64]), &device);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .per_block([32], ScaleDtype::F32);

    should_dequantize_transposed(tensor, scheme);
}

#[test]
fn should_dequantize_symmetric_per_block_int8_permuted_2x8x16() {
    let device = Default::default();

    let values = (0..256)
        .map(|value| value as f32 / 256.)
        .collect::<Vec<_>>();
    let tensor = TestTensor::<3>::from_data(TensorData::new(values, [2, 8, 16]), &device);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .per_block([1, 2, 16], ScaleDtype::F32);
    let expected = tensor.clone().permute([1, 2, 0]);
    let output = tensor
        .quantize_dynamic(&scheme)
        .permute([1, 2, 0])
        .dequantize();

    expected.into_data().assert_approx_eq::<FloatElem>(
        &output.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

#[test]
fn should_dequantize_symmetric_per_block_int8_permuted_packed_axis_first() {
    let device = Default::default();

    let values = (0..256)
        .map(|value| value as f32 / 256.)
        .collect::<Vec<_>>();
    let tensor = TestTensor::<3>::from_data(TensorData::new(values, [2, 8, 16]), &device);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .per_block([1, 2, 16], ScaleDtype::F32);
    let expected = tensor.clone().permute([2, 0, 1]);
    let output = tensor
        .quantize_dynamic(&scheme)
        .permute([2, 0, 1])
        .dequantize();

    expected.into_data().assert_approx_eq::<FloatElem>(
        &output.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

#[test]
fn should_quantize_symmetric_int8_permuted_batch_dims() {
    let device = Default::default();

    let tensor = TestTensorInt::arange(0..2048, &device)
        .float()
        .div_scalar(2048.)
        .reshape([2, 4, 8, 32]);

    // Permute [0,1,2,3] -> [1,2,0,3]
    // This rearranges batch dims but keeps packed dim in place
    let tensor_permuted = tensor.clone().permute([1, 2, 0, 3]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    let output = tensor_permuted
        .quantize_dynamic(&scheme)
        .dequantize()
        .permute([2, 0, 1, 3]); // reverse permutation

    tensor.into_data().assert_approx_eq::<FloatElem>(
        &output.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

#[test]
fn should_quantize_symmetric_two_level_f16_block_scales() {
    let device = Default::default();

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &device)
        .float()
        .div_scalar(256.)
        .reshape([16, 16]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .per_block([2, 16], ScaleDtype::F16)
        .per_tensor(ScaleDtype::F32);

    let quantized = input.clone().quantize_dynamic(&scheme);
    let direct = quantized.clone().dequantize().into_data();

    let reloaded = TestTensor::<2>::from_data(quantized.into_data(), &device);
    let round_tripped = reloaded.dequantize().into_data();

    round_tripped.assert_eq(&direct, true);
    direct.assert_approx_eq(&input.into_data(), Tolerance::<f32>::rel_abs(1e-2, 1e-2));
}
