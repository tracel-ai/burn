use super::*;
use burn_tensor::TensorData;

#[test]
fn test_any() {
    let tensor = TestTensorInt::<2>::from([[0, 0, 0], [1, -1, 0]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([true]);
    data_expected.assert_eq(&data_actual, false);

    let tensor = TestTensorInt::<2>::from([[0, 0, 0], [0, 0, 0]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([false]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_any_dim() {
    let tensor = TestTensorInt::<2>::from([[0, 0, 0], [1, -1, 0]]);
    let data_actual = tensor.any_dim(1).into_data();
    let data_expected = TensorData::from([[false], [true]]);
    data_expected.assert_eq(&data_actual, false);
}
