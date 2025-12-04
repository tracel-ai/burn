use super::*;
use crate::qtensor::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn clamp_min() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.clamp_min(2.0);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn clamp_max() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.clamp_max(2.0);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[0.0, 1.0, 2.0], [2.0, 2.0, 2.0]]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn clamp_min_max() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.clamp(1.0, 4.0);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[1.0, 1.0, 2.0], [3.0, 4.0, 4.0]]),
            Tolerance::absolute(1e-1),
        );
}
