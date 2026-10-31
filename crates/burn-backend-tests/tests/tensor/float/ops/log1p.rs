use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn log1p_preserves_small_inputs() {
    use burn_tensor::ElementConversion;
    let half = <FloatElem as burn_tensor::Element>::dtype() == burn_tensor::DType::F16;
    let small = if half { 1e-4 } else { 1e-8 };
    let values: Vec<FloatElem> = [small, -small, 1e-4, -1e-4, 0.005, -0.005]
        .into_iter()
        .map(|value| value.elem())
        .collect();
    let expected: Vec<f64> = values
        .iter()
        .map(|value| value.elem::<f64>().ln_1p())
        .collect();
    let tensor = TestTensor::<1>::from_data(TensorData::new(values, [6]), &Default::default());
    tensor
        .log1p()
        .into_data()
        .convert::<f64>()
        .assert_approx_eq::<f64>(
            &TensorData::new(expected, [6]),
            Tolerance::relative(if half { 1e-2 } else { 1e-5 }),
        );
}

#[test]
fn should_support_exp_log1p() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.log1p();
    let expected = TensorData::from([
        [0.0, core::f32::consts::LN_2, 1.09861],
        [1.38629, 1.60944, 1.79176],
    ]);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::default().set_half_precision_relative(1e-3),
    );
}
