use crate::qtensor::*;
use crate::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_sub_ops() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]);

    let output = tensor_1 - tensor_2;
    let expected = TensorData::from([[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]]);

    
    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn test_sub_broadcast() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);

    let output = tensor_1 - tensor_2;
    let expected = TensorData::from([[-3.0, -3.0, -3.0], [-6.0, -6.0, -6.0]]);

    
    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_sub_scalar_ops() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let scalar = 2.0;

    let output = tensor - scalar;
    let expected = TensorData::from([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}
