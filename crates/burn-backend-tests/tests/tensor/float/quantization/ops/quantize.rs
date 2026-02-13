use super::*;
use alloc::{vec, vec::Vec};
use burn_tensor::quantization::{
    QParams, QTensorPrimitive, QuantLevel, QuantScheme, QuantStore, QuantValue,
    QuantizationParameters, QuantizedBytes,
};
use burn_tensor::{DType, Element, TensorData};
use burn_tensor::{Tolerance, ops::QuantizedTensor};

fn get_q_params(data: TensorData) -> QParams<Vec<f32>> {
    let num_elements = data.num_elements();
    let scheme = if let DType::QFloat(scheme) = data.dtype {
        scheme
    } else {
        unreachable!()
    };
    let q_bytes = QuantizedBytes {
        bytes: data.into_bytes(),
        scheme,
        num_elements,
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
    let tensor = TestTensor::<1>::from_floats([-1.8, -1.0, 0.0, 0.5], &device);
    let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);
    let qparams = QuantizationParameters {
        scales: TestTensor::from_floats([0.014_173_228], &device),
    };

    let x_q = tensor.clone().quantize(&scheme, qparams);

    let x_q_data = x_q.to_data();
    let expected = TensorData::quantized(
        vec![-127i8, -71, 0, 35],
        [4],
        scheme.with_store(QuantStore::Native),
        &[0.014_173_228], // scale
    );

    // Values equality
    x_q_data.assert_eq(&expected, false);

    // Quantization parameters check
    let qparams = get_q_params(x_q_data);
    let expected = get_q_params(expected);
    assert_eq!(qparams.scales.len(), 1);
    // TODO: check scales
    assert_eq!(qparams.scales, expected.scales);

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
    let tensor = TestTensor::<1>::from_floats([5., 0., 4., -12.7], &device);
    let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);

    let x_q = tensor.quantize_dynamic(&scheme);

    let expected = TensorData::quantized(
        vec![50i8, 0, 40, -127],
        [4],
        scheme.with_store(QuantStore::Native),
        &[0.1], // scale
    );

    x_q.into_data().assert_eq(&expected, false);
}

#[test]
fn should_quantize_dequantize_symmetric_single_with_transform() {
    let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);
    let input = TestTensorInt::<1>::arange(0..32, &Default::default()).float();

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
    let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &Default::default())
        .float()
        .div_scalar(256.)
        .reshape([16, 16]);

    let output = input.clone().quantize_dynamic(&scheme);
    let output = output.dequantize();

    output.into_data().assert_approx_eq::<FloatElem>(
        &input.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

#[test]
fn should_quantize_dequantize_symmetric_per_block_arange_16x16() {
    let scheme = QuantizedTensor::<TestBackend>::default_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block([2, 16]));

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &Default::default())
        .float()
        .div_scalar(256.)
        .reshape([16, 16]);

    let output = input.clone().quantize_dynamic(&scheme);
    let output = output.dequantize();

    output.into_data().assert_approx_eq::<FloatElem>(
        &input.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

fn should_quantize_transposed<const D: usize>(tensor: Tensor<TestBackend, D>, scheme: QuantScheme) {
    let tensor_t = tensor.clone().transpose();

    let output = tensor_t.quantize_dynamic(&scheme).dequantize().transpose();

    tensor.into_data().assert_approx_eq::<FloatElem>(
        &output.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

#[test]
fn should_quantize_symmetric_int8_transposed_8x32() {
    let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);

    let tensor = TestTensorInt::arange(0..256, &Default::default())
        .float()
        .div_scalar(256.)
        .reshape([8, 32]);
    should_quantize_transposed(tensor, scheme);
}

#[test]
fn should_quantize_symmetric_int8_transposed_48x64() {
    let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);

    let tensor = TestTensorInt::arange(0..3072, &Default::default())
        .float()
        .div_scalar(3072.)
        .reshape([48, 64]);
    should_quantize_transposed(tensor, scheme);
}

#[test]
fn should_quantize_symmetric_per_block_int8_transposed_32x64() {
    let scheme = QuantizedTensor::<TestBackend>::default_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block([32]));

    let tensor = TestTensorInt::arange(0..2048, &Default::default())
        .float()
        .div_scalar(2048.)
        .reshape([32, 64]);
    should_quantize_transposed(tensor, scheme);
}

#[test]
fn should_quantize_symmetric_int8_permuted_batch_dims() {
    let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);

    let tensor = TestTensorInt::arange(0..2048, &Default::default())
        .float()
        .div_scalar(2048.)
        .reshape([2, 4, 8, 32]);

    // Permute [0,1,2,3] -> [1,2,0,3]
    // This rearranges batch dims but keeps packed dim in place
    let tensor_permuted = tensor.clone().permute([1, 2, 0, 3]);

    let output = tensor_permuted
        .quantize_dynamic(&scheme)
        .dequantize()
        .permute([2, 0, 1, 3]); // reverse permutation

    tensor.into_data().assert_approx_eq::<FloatElem>(
        &output.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}
