use super::*;
use crate::qtensor::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_powf_ops() {
    let tensor = QTensor::<TestBackend, 2>::int8([[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_pow = QTensor::<TestBackend, 2>::int8([[1.0, 1.0, 2.0], [3.0, 4.0, 2.0]]);

    let output = tensor.powf(tensor_pow);
    let expected = TensorData::from([[1.0, 1.0, 4.0], [27.0, 256.0, 25.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(4e-2, 1e-2));
}

#[test]
fn should_support_neg_power() {
    let tensor = QTensor::<TestBackend, 2>::int8([[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_pow = QTensor::<TestBackend, 2>::int8([[-0.95, -0.67, -0.45], [-0.24, -0.5, -0.6]]);

    let output = tensor.powf(tensor_pow);
    let expected = TensorData::from([[1., 1., 0.73204285], [0.76822936, 0.5, 0.38073079]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(4e-2, 1e-2));
}

#[test]
fn should_support_neg_values_with_even_power() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]);
    let tensor_pow = QTensor::<TestBackend, 2>::int8([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);

    let output = tensor.powf(tensor_pow);
    let expected = TensorData::from([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(4e-2, 1e-2));
}

#[test]
fn should_support_neg_values_with_odd_power() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, -1.0, -2.0], [-3.0, -4.0, -4.0]]);
    let tensor_pow = QTensor::<TestBackend, 2>::int8([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]);

    let output = tensor.powf(tensor_pow);
    let expected = TensorData::from([[0.0, -1.0, -8.0], [-27.0, -64.0, -64.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(4e-2, 1e-2));
}
