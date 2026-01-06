use super::*;
use alloc::{vec, vec::Vec};
use burn_tensor::{Tensor, TensorData};

#[test]
fn should_support_stack_ops_2d_dim0() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_data([[1.0, 2.0, 3.0]], &device);
    let tensor_2 = TestTensor::from_data([[4.0, 5.0, 6.0]], &device);

    let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 0);
    let expected = TensorData::from([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_stack_ops_2d_dim1() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_data([[1.0, 2.0, 3.0]], &device);
    let tensor_2 = TestTensor::from_data([[4.0, 5.0, 6.0]], &device);

    let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 1);
    let expected = TensorData::from([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_stack_ops_3d() {
    let device = Default::default();
    let tensor_1 = TestTensor::<3>::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]], &device);
    let tensor_2 = TestTensor::from_data([[[4.0, 5.0, 6.0]], [[4.1, 5.1, 6.1]]], &device);

    let output = Tensor::stack::<4>(vec![tensor_1, tensor_2], 0);
    let expected = TensorData::from([
        [[[1.0000, 2.0000, 3.0000]], [[1.1000, 2.1000, 3.1000]]],
        [[[4.0000, 5.0000, 6.0000]], [[4.1000, 5.1000, 6.1000]]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic]
fn should_panic_when_dimensions_are_not_the_same() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], &device);
    let tensor_2 = TestTensor::from_data([[4.0, 5.0]], &device);

    let _output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 0);
}

#[test]
#[should_panic]
fn should_panic_when_list_of_vectors_is_empty() {
    let tensors: Vec<TestTensor<2>> = vec![];
    let _output = Tensor::stack::<3>(tensors, 0);
}

#[test]
#[should_panic]
fn should_panic_when_stack_exceeds_dimension() {
    let device = Default::default();
    let tensor_1 = TestTensor::<3>::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]], &device);
    let tensor_2 = TestTensor::from_data([[[4.0, 5.0, 6.0]]], &device);

    let _output = Tensor::stack::<4>(vec![tensor_1, tensor_2], 3);
}
