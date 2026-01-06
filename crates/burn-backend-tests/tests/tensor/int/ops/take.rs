use super::*;
use burn_tensor::TensorData;

#[test]
fn should_take_int_tensor() {
    // Test take with integer tensors
    let device = Default::default();
    let tensor = TestTensorInt::<2>::from_data([[10, 20, 30], [40, 50, 60]], &device);
    let indices = TestTensorInt::<1>::from_data([1, 0], &device);

    let output = tensor.take::<1, 2>(0, indices);
    let expected = TensorData::from([[40, 50, 60], [10, 20, 30]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_take_int_tensor_with_2d_indices() {
    // Test take with integer tensors - output will be 3D
    let device = Default::default();
    let tensor = TestTensorInt::<2>::from_data([[10, 20, 30], [40, 50, 60], [70, 80, 90]], &device);

    // 2D indices - shape [2, 2]
    let indices = TestTensorInt::<2>::from_data([[0, 2], [2, 1]], &device);
    let output = tensor.take::<2, 3>(0, indices);

    // Expected: shape [2, 2, 3]
    let expected = TensorData::from([[[10, 20, 30], [70, 80, 90]], [[70, 80, 90], [40, 50, 60]]]);

    output.into_data().assert_eq(&expected, false);
}
