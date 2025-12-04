use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_erf_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.erf();
    let expected = TensorData::from([[0.0000, 0.8427, 0.99532], [0.99998, 1.0000, 1.0000]]);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::default().set_half_precision_absolute(2e-3),
    );
}

#[test]
fn should_support_erf_ops_with_negative_number() {
    let data = TensorData::from([[-0.056, -0.043, -0.089], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.erf();
    let expected = TensorData::from([
        [-0.06312324, -0.048490416, -0.10016122],
        [0.99998, 1.0000, 1.0000],
    ]);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::default().set_half_precision_absolute(3e-3),
    );
}
