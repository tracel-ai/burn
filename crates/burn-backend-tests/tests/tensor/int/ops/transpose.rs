use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_transpose_ops_int() {
    let tensor = TestTensorInt::<3>::from_data(
        [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
        &Default::default(),
    );

    let output = tensor.transpose();
    let expected = TensorData::from([[[0, 3], [1, 4], [2, 5]], [[6, 9], [7, 10], [8, 11]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_swap_dims_int() {
    let tensor = TestTensorInt::<3>::from_data(
        [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
        &Default::default(),
    );

    let output = tensor.swap_dims(0, 2);
    let expected = TensorData::from([[[0, 6], [3, 9]], [[1, 7], [4, 10]], [[2, 8], [5, 11]]]);

    output.into_data().assert_eq(&expected, false);
}
