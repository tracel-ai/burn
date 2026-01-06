use super::*;
use burn_tensor::TensorData;

#[test]
fn test_any() {
    let tensor = TestTensorBool::<2>::from([[false, false, false], [true, true, false]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([true]);
    data_expected.assert_eq(&data_actual, false);

    let tensor = TestTensorBool::<2>::from([[false, false, false], [false, false, false]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([false]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_any_dim() {
    let tensor = TestTensorBool::<2>::from([[false, false, false], [true, true, false]]);
    let data_actual = tensor.any_dim(1).into_data();
    let data_expected = TensorData::from([[false], [true]]);
    data_expected.assert_eq(&data_actual, false);
}
