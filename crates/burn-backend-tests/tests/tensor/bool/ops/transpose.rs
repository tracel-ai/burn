use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_transpose_bool() {
    let tensor = TestTensorBool::<3>::from_data(
        [
            [[false, true, false], [false, false, false]],
            [[false, false, true], [false, false, true]],
        ],
        &Default::default(),
    );

    let output = tensor.transpose();
    let expected = TensorData::from([
        [[false, false], [true, false], [false, false]],
        [[false, false], [false, false], [true, true]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_swap_dims_bool() {
    let tensor = TestTensorBool::<3>::from_data(
        [
            [[false, true, false], [false, false, false]],
            [[false, false, true], [false, false, true]],
        ],
        &Default::default(),
    );

    let output = tensor.swap_dims(0, 2);
    let expected = TensorData::from([
        [[false, false], [false, false]],
        [[true, false], [false, false]],
        [[false, true], [false, true]],
    ]);

    output.into_data().assert_eq(&expected, false);
}
