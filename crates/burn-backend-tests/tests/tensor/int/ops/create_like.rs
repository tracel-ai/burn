use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_zeros_like() {
    let tensor = TestTensorInt::<3>::from([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]);

    let tensor = tensor.zeros_like();
    let expected = TensorData::from([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]);

    tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_ones_like() {
    let tensor = TestTensorInt::<3>::from([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]);

    let tensor = tensor.ones_like();
    let expected = TensorData::from([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]);

    tensor.into_data().assert_eq(&expected, false);
}
