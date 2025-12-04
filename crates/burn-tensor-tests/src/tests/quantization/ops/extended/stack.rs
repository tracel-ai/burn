use crate::qtensor::*;
use crate::*;
use alloc::vec;
use burn_tensor::Tensor;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_stack_ops_2d_dim0() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[1.0, 2.0, 3.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[4.0, 5.0, 6.0]]);

    let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 0);
    let expected = TensorData::from([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]);

    
    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_stack_ops_2d_dim1() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[1.0, 2.0, 3.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[4.0, 5.0, 6.0]]);

    let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 1);
    let expected = TensorData::from([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);

    
    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_stack_ops_3d() {
    let tensor_1 = QTensor::<TestBackend, 3>::int8([[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]]);
    let tensor_2 = QTensor::<TestBackend, 3>::int8([[[4.0, 5.0, 6.0]], [[6.0, 5.0, 4.0]]]);

    let output = Tensor::stack::<4>(vec![tensor_1, tensor_2], 0);
    let expected = TensorData::from([
        [[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]],
        [[[4.0, 5.0, 6.0]], [[6.0, 5.0, 4.0]]],
    ]);

    
    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
#[should_panic]
fn should_panic_when_dimensions_are_not_the_same() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[1.0, 2.0, 3.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[4.0, 5.0]]);

    let _output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 0);
}

#[test]
#[should_panic]
fn should_panic_when_stack_exceeds_dimension() {
    let tensor_1 = QTensor::<TestBackend, 3>::int8([[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]]);
    let tensor_2 = QTensor::<TestBackend, 3>::int8([[[4.0, 5.0, 6.0]]]);

    let _output = Tensor::stack::<4>(vec![tensor_1, tensor_2], 3);
}
