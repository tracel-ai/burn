use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_bool_empty() {
    let shape = [2, 2];
    let tensor = TestTensorBool::<2>::empty(shape, &Default::default());
    assert_eq!(tensor.shape(), shape.into())
}

#[test]
fn should_support_bool_zeros() {
    let shape = [2, 2];
    let tensor = TestTensorBool::<2>::zeros(shape, &Default::default());
    assert_eq!(tensor.shape(), shape.into());

    tensor
        .into_data()
        .assert_eq(&TensorData::from([[false, false], [false, false]]), false);
}

#[test]
fn should_support_bool_ones() {
    let shape = [2, 2];
    let tensor = TestTensorBool::<2>::ones(shape, &Default::default());
    assert_eq!(tensor.shape(), shape.into());

    tensor
        .into_data()
        .assert_eq(&TensorData::from([[true, true], [true, true]]), false);
}
