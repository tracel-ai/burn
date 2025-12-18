use super::*;
use burn_tensor::TensorData;

#[test]
fn test_bool_and() {
    let tensor1 = TestTensorBool::<2>::from([[false, true, false], [true, false, true]]);
    let tensor2 = TestTensorBool::<2>::from([[true, true, false], [false, false, true]]);
    let data_actual = tensor1.bool_and(tensor2).into_data();
    let data_expected = TensorData::from([[false, true, false], [false, false, true]]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_bool_or() {
    let tensor1 = TestTensorBool::<2>::from([[false, true, false], [true, false, true]]);
    let tensor2 = TestTensorBool::<2>::from([[true, true, false], [false, false, true]]);
    let data_actual = tensor1.bool_or(tensor2).into_data();
    let data_expected = TensorData::from([[true, true, false], [true, false, true]]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_bool_xor() {
    let tensor1 = TestTensorBool::<2>::from([[false, true, false], [true, false, true]]);
    let tensor2 = TestTensorBool::<2>::from([[true, true, false], [false, false, true]]);
    let data_actual = tensor1.bool_xor(tensor2).into_data();
    let data_expected = TensorData::from([[true, false, false], [true, false, false]]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_bool_or_vec() {
    let device = Default::default();
    let tensor1 = TestTensorBool::<1>::full([256], 0, &device);
    let tensor2 = TestTensorBool::<1>::full([256], 1, &device);
    let data_actual = tensor1.bool_or(tensor2).into_data();
    let data_expected = TensorData::from([true; 256]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_bool_and_vec() {
    let device = Default::default();
    let tensor1 = TestTensorBool::<1>::full([256], 0, &device);
    let tensor2 = TestTensorBool::<1>::full([256], 1, &device);
    let data_actual = tensor1.bool_and(tensor2).into_data();
    let data_expected = TensorData::from([false; 256]);
    data_expected.assert_eq(&data_actual, false);
}
