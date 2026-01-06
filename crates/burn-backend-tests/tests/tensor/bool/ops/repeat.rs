use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_bool_repeat_ops_one_dimension() {
    let data = TensorData::from([[true, false, false]]);
    let tensor = TestTensorBool::<2>::from_data(data, &Default::default());

    let output = tensor.repeat(&[4, 1, 1]);
    let expected = TensorData::from([
        [true, false, false],
        [true, false, false],
        [true, false, false],
        [true, false, false],
    ]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_bool_repeat_on_many_dimension() {
    let data = TensorData::from([
        [[false, true], [true, false]],
        [[true, true], [false, false]],
    ]);
    let tensor = TestTensorBool::<3>::from_data(data, &Default::default());

    let output = tensor.repeat(&[2, 3, 2]);
    let expected = TensorData::from([
        [
            [false, true, false, true],
            [true, false, true, false],
            [false, true, false, true],
            [true, false, true, false],
            [false, true, false, true],
            [true, false, true, false],
        ],
        [
            [true, true, true, true],
            [false, false, false, false],
            [true, true, true, true],
            [false, false, false, false],
            [true, true, true, true],
            [false, false, false, false],
        ],
        [
            [false, true, false, true],
            [true, false, true, false],
            [false, true, false, true],
            [true, false, true, false],
            [false, true, false, true],
            [true, false, true, false],
        ],
        [
            [true, true, true, true],
            [false, false, false, false],
            [true, true, true, true],
            [false, false, false, false],
            [true, true, true, true],
            [false, false, false, false],
        ],
    ]);

    output.into_data().assert_eq(&expected, false);
}
