use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_bool_repeat_ops() {
    let data = TensorData::from([[true, false, false]]);
    let tensor = TestTensorBool::<2>::from_data(data, &Default::default());

    let output = tensor.repeat_dim(0, 4);
    let expected = TensorData::from([
        [true, false, false],
        [true, false, false],
        [true, false, false],
        [true, false, false],
    ]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_bool_repeat_on_dims_larger_than_1() {
    let data = TensorData::from([
        [[false, true], [true, false]],
        [[true, true], [false, false]],
    ]);
    let tensor = TestTensorBool::<3>::from_data(data, &Default::default());

    let output = tensor.repeat_dim(1, 2);
    let expected = TensorData::from([
        [[false, true], [true, false], [false, true], [true, false]],
        [[true, true], [false, false], [true, true], [false, false]],
    ]);

    output.into_data().assert_eq(&expected, false);
}
