use super::qtensor::*;
use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_erf_ops() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.erf();
    let expected = TensorData::from([[0.0000, 0.8427, 0.9953], [1.0000, 1.0000, 1.0000]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_erf_ops_with_negative_number() {
    let tensor = QTensor::<TestBackend, 2>::int8([[-0.056, -0.043, -0.089], [3.0, 4.0, 5.0]]);

    let output = tensor.erf();
    let expected = TensorData::from([
        [-0.06312324, -0.048490416, -0.10016122],
        [1.0000, 1.0000, 1.0000],
    ]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}
