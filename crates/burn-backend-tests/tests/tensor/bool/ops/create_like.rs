use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_zeros_like() {
    let tensor = TestTensorBool::<3>::from([
        [[false, true, false], [true, true, true]],
        [[false, false, false], [true, true, false]],
    ]);

    let tensor = tensor.zeros_like();
    let expected = TensorData::from([
        [[false, false, false], [false, false, false]],
        [[false, false, false], [false, false, false]],
    ]);

    tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_ones_like() {
    let tensor = TestTensorBool::<3>::from([
        [[false, true, false], [true, true, true]],
        [[false, false, false], [true, true, false]],
    ]);

    let tensor = tensor.ones_like();
    let expected = TensorData::from([
        [[true, true, true], [true, true, true]],
        [[true, true, true], [true, true, true]],
    ]);

    tensor.into_data().assert_eq(&expected, false);
}
