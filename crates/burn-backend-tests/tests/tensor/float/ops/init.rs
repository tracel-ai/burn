use super::*;
use burn_tensor::{DType, TensorData};

#[test]
fn should_support_float_empty() {
    let shape = [2, 2];
    let tensor = TestTensor::<2>::empty(shape, &Default::default());
    assert_eq!(tensor.shape(), shape.into())
}

#[test]
fn should_support_float_empty_options() {
    let shape = [2, 2];
    let tensor = TestTensor::<2>::empty(shape, (&Default::default(), DType::F32));
    assert_eq!(tensor.shape(), shape.into())
}

#[test]
fn should_support_float_zeros() {
    let shape = [2, 2];
    let tensor = TestTensor::<2>::zeros(shape, &Default::default());
    assert_eq!(tensor.shape(), shape.into());

    tensor
        .into_data()
        .assert_eq(&TensorData::from([[0., 0.], [0., 0.]]), false);
}

#[test]
fn should_support_float_zeros_options() {
    let shape = [2, 2];
    let tensor = TestTensor::<2>::zeros(shape, (&Default::default(), DType::F32));
    assert_eq!(tensor.shape(), shape.into());
    assert_eq!(tensor.dtype(), DType::F32);

    tensor
        .into_data()
        .assert_eq(&TensorData::from([[0., 0.], [0., 0.]]), false);
}

#[test]
fn should_support_float_ones() {
    let shape = [2, 2];
    let tensor = TestTensor::<2>::ones(shape, &Default::default());
    assert_eq!(tensor.shape(), shape.into());

    tensor
        .into_data()
        .assert_eq(&TensorData::from([[1., 1.], [1., 1.]]), false);
}

#[test]
fn should_support_float_ones_options() {
    let shape = [2, 2];
    let tensor = TestTensor::<2>::ones(shape, (&Default::default(), DType::F32));
    assert_eq!(tensor.shape(), shape.into());
    assert_eq!(tensor.dtype(), DType::F32);

    tensor
        .into_data()
        .assert_eq(&TensorData::from([[1., 1.], [1., 1.]]), false);
}
