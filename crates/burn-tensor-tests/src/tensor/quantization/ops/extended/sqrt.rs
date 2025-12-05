use super::qtensor::*;
use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use core::f32::consts::SQRT_2;

#[test]
fn should_support_sqrt_ops() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.sqrt();
    let expected = TensorData::from([[0.0, 1.0, SQRT_2], [1.73205, 2.0, 2.2360]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}
