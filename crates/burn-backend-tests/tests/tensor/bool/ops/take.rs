use super::*;
use burn_tensor::TensorData;

#[test]
fn should_take_bool_tensor() {
    // Test take with boolean tensors
    let device = Default::default();
    let tensor = TestTensorBool::<2>::from_data([[true, false], [false, true]], &device);
    let indices = TestTensorInt::<1>::from_data([1, 0], &device);

    let output = tensor.take::<1, 2>(0, indices);
    let expected = TensorData::from([[false, true], [true, false]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_take_bool_tensor_with_2d_indices() {
    // Test take with boolean tensors - output will be 3D
    let device = Default::default();
    let tensor = TestTensorBool::<2>::from_data(
        [
            [true, false, true],
            [false, true, false],
            [true, true, false],
        ],
        &device,
    );

    // 2D indices - shape [2, 2]
    let indices = TestTensorInt::<2>::from_data([[0, 2], [1, 0]], &device);
    let output = tensor.take::<2, 3>(0, indices);

    // Expected: shape [2, 2, 3]
    let expected = TensorData::from([
        [[true, false, true], [true, true, false]],
        [[false, true, false], [true, false, true]],
    ]);

    output.into_data().assert_eq(&expected, false);
}
