use super::*;
use burn_tensor::{DType, TensorData};

#[test]
fn test_data_full() {
    let tensor = TensorData::full([2, 3], 2.0);

    tensor.assert_eq(&TensorData::from([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), false);
}

#[test]
fn test_tensor_full() {
    let device = Default::default();
    let tensor = TestTensor::<2>::full([2, 3], 2.1, &device);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([[2.1, 2.1, 2.1], [2.1, 2.1, 2.1]]), false);
}

#[test]
fn test_tensor_full_options() {
    let tensor = TestTensor::<2>::full([2, 3], 2.1, (&Default::default(), DType::F32));
    assert_eq!(tensor.dtype(), DType::F32);

    tensor
        .into_data()
        .assert_eq(&TensorData::from([[2.1, 2.1, 2.1], [2.1, 2.1, 2.1]]), false);
}
