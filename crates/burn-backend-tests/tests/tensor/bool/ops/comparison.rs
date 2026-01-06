use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_bool_equal() {
    let data_1 = TensorData::from([[false, true, true], [true, false, true]]);
    let data_2 = TensorData::from([[false, false, true], [false, true, true]]);
    let device = Default::default();
    let tensor_1 = TestTensorBool::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorBool::<2>::from_data(data_2, &device);

    let data_actual_cloned = tensor_1.clone().equal(tensor_2.clone());
    let data_actual_inplace = tensor_1.equal(tensor_2);

    let data_expected = TensorData::from([[true, false, true], [false, false, true]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn should_support_bool_not_equal() {
    let data_1 = TensorData::from([[false, true, true], [true, false, true]]);
    let data_2 = TensorData::from([[false, false, true], [false, true, true]]);
    let device = Default::default();
    let tensor_1 = TestTensorBool::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorBool::<2>::from_data(data_2, &device);

    let data_actual_cloned = tensor_1.clone().not_equal(tensor_2.clone());
    let data_actual_inplace = tensor_1.not_equal(tensor_2);

    let data_expected = TensorData::from([[false, true, false], [true, true, false]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn should_support_bool_not() {
    let data_1 = TensorData::from([[false, true, true], [true, true, false]]);
    let tensor_1 = TestTensorBool::<2>::from_data(data_1, &Default::default());

    let data_actual_cloned = tensor_1.clone().bool_not();
    let data_actual_inplace = tensor_1.bool_not();

    let data_expected = TensorData::from([[true, false, false], [false, false, true]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_bool_equal_elem() {
    let tensor_1 = TestTensorBool::<2>::from([[true, false, true], [false, true, false]]);

    let data_actual_cloned = tensor_1.clone().equal_elem(false);
    let data_actual_inplace = tensor_1.equal_elem(false);

    let data_expected = TensorData::from([[false, true, false], [true, false, true]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_bool_not_equal_elem() {
    let tensor_1 = TestTensorBool::<2>::from([[true, false, true], [false, true, false]]);

    let data_actual_cloned = tensor_1.clone().not_equal_elem(true);
    let data_actual_inplace = tensor_1.not_equal_elem(true);

    let data_expected = TensorData::from([[false, true, false], [true, false, true]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}
