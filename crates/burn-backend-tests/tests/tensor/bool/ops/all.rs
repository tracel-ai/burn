use super::*;
use burn_tensor::TensorData;

#[test]
fn test_all() {
    let tensor = TestTensorBool::<2>::from([[false, true, false], [true, true, true]]);
    let data_actual = tensor.all().into_data();
    let data_expected = TensorData::from([false]);
    data_expected.assert_eq(&data_actual, false);

    let tensor = TestTensorBool::<2>::from([[true, true, true], [true, true, true]]);
    let data_actual = tensor.all().into_data();
    let data_expected = TensorData::from([true]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_all_dim() {
    let tensor = TestTensorBool::<2>::from([[false, true, false], [true, true, true]]);
    let data_actual = tensor.all_dim(1).into_data();
    let data_expected = TensorData::from([[false], [true]]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_all_with_bool_from_lower_equal() {
    let tensor1 = TestTensor::<2>::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]) + 1e-6;
    let tensor2 = TestTensor::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]) + 1e-6;

    let ge = tensor1.lower_equal(tensor2);
    let all = ge.clone().all();

    TensorData::from([true]).assert_eq(&all.clone().into_data(), false);
}
