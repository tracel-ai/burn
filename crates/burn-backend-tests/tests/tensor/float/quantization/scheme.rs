use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{
    Device, Element, FloatDType, TensorData,
    quantization::{CalibrationRange, QuantLevel, QuantParam, QuantValue, compute_q_params},
};

#[test]
fn per_tensor_symmetric_int8() {
    let device = Default::default();
    let range = CalibrationRange {
        min: TestTensor::<1>::from_data([0.5], &device),
        max: TestTensor::<1>::from_data([1.8], &device),
    };
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    let qparams = compute_q_params(&scheme, range);

    qparams
        .scales
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([0.014_173_23]), Tolerance::default());
}

#[test]
fn per_block_symmetric_int8() {
    let device = Default::default();
    let range = CalibrationRange {
        min: TestTensor::<1>::from_data([-1.8, -0.5, 0.01, -0.04], &device),
        max: TestTensor::<1>::from_data([0.5, 1.8, 0.04, -0.01], &device),
    };
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block([4]));

    let qparams = compute_q_params(&scheme, range);

    qparams.scales.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([0.014_173_23, 0.014_173_23, 0.000_314_96, 0.000_314_96]),
        Tolerance::default(),
    );
}

#[test]
fn block_tensor_symmetric_int8() {
    let device = Default::default();
    let min = TestTensor::<1>::from_data([-1.8, -0.5, 0.01, -0.04], &device);
    let max = TestTensor::<1>::from_data([0.5, 1.8, 0.04, -0.01], &device);
    let range = || CalibrationRange {
        min: min.clone(),
        max: max.clone(),
    };

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    let one_level = scheme.with_level(QuantLevel::block([4]));
    let two_level = scheme
        .with_level(QuantLevel::block_tensor([4], QuantParam::F32))
        .with_param(QuantParam::UE4M3);

    let expected = compute_q_params(&one_level, range()).scales.into_data();
    let qparams = compute_q_params(&two_level, range());
    let global = qparams
        .global
        .expect("a two-level scheme should produce a per-tensor scale");

    // The largest block scale is pushed to the top of what ue4m3 can hold, which is the point of
    // splitting the scale in two.
    qparams
        .scales
        .clone()
        .max()
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([448.0]), Tolerance::default());

    // The global is f32 whatever the element type is, hence the cast.
    qparams
        .scales
        .mul(global.cast(FloatDType::from(FloatElem::dtype())))
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

/// A tensor with nothing in it divides `0` by `448`, and dividing the block scales by that
/// quotient is where a zero becomes a NaN.
#[test]
fn block_tensor_symmetric_int8_all_zero() {
    let device = Device::default();
    let zeros = TestTensor::<1>::zeros([4], &device);
    let range = CalibrationRange {
        min: zeros.clone(),
        max: zeros,
    };

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block_tensor([4], QuantParam::F32))
        .with_param(QuantParam::UE4M3);

    let qparams = compute_q_params(&scheme, range);
    let global: f32 = qparams
        .global
        .expect("a two-level scheme should produce a per-tensor scale")
        .into_scalar();
    let scales: Vec<f32> = qparams.scales.into_data().iter::<f32>().collect();

    assert!(
        global > 0.0,
        "a per-tensor scale of {global} leaves nothing to divide the block scales by"
    );
    assert!(
        scales.iter().all(|scale| scale.is_finite()),
        "block scales should stay finite for an empty tensor, got {scales:?}"
    );
}

#[test]
fn quant_scheme_should_inhibit_by_default() {
    let device = Device::default();
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    let tensor_1 = TestTensor::<2>::from_data(
        [[1.0, 6.35, 0., 0.], [2.0, 3.0, 0., 0.], [1.0, 3.0, 0., 0.]],
        &device,
    )
    .quantize_dynamic(&scheme);
    let _tensor_2 = TestTensor::<2>::from_data(
        [
            [4.0, 8.0, 12.7, 0.],
            [2.0, 3.0, 6.0, 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
        &device,
    )
    .quantize_dynamic(&scheme);

    // let tensor_3 = tensor_1.clone().matmul(tensor_2);
    // assert_eq!(tensor_3.to_data().dtype, FloatElem::dtype());

    let tensor_4 = tensor_1.add_scalar(1.);
    assert_eq!(tensor_4.to_data().dtype, FloatElem::dtype());
}
