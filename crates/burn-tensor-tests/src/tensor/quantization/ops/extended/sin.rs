use super::*;
use crate::qtensor::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_sin_ops() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.sin();
    let expected = TensorData::from([[0.0, 0.8414, 0.9092], [0.1411, -0.7568, -0.9589]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}
