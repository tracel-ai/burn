use crate::*;
use burn_tensor::TensorData;

#[test]
fn clamp_min() {
    let device = Default::default();
    // test float tensor
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &device);

    let output = tensor.clamp_min(2.0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]]), false);

    // test int tensor
    let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let tensor = TestTensorInt::<2>::from_data(data, &device);
    let output = tensor.clamp_min(2);

    output
        .into_data()
        .assert_eq(&TensorData::from([[2, 2, 2], [3, 4, 5]]), false);
}

#[test]
fn clamp_max() {
    let device = Default::default();
    // test float tensor
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &device);

    let output = tensor.clamp_max(2.0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0.0, 1.0, 2.0], [2.0, 2.0, 2.0]]), false);

    // test int tensor
    let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let tensor = TestTensorInt::<2>::from_data(data, &device);
    let output = tensor.clamp_max(4);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0, 1, 2], [3, 4, 4]]), false);
}

#[test]
fn clamp_min_max() {
    let device = Default::default();
    // test float tensor
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &device);
    let output = tensor.clamp(1.0, 4.0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1.0, 1.0, 2.0], [3.0, 4.0, 4.0]]), false);

    // test int tensor
    let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let tensor = TestTensorInt::<2>::from_data(data, &device);
    let output = tensor.clamp(1, 4);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 1, 2], [3, 4, 4]]), false);
}

#[test]
fn clamp_min_max_vec_should_compile() {
    let input = TestTensor::<2>::ones([2, 4], &Default::default());
    let output = input.clamp(0., 0.5);

    output.into_data().assert_eq(
        &TensorData::from([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),
        false,
    );
}
