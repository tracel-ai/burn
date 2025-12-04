use super::*;
use crate::qtensor::*;
use burn_tensor::TensorData;

#[test]
fn test_equal() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 1.0], [3.0, 5.0, 4.0]]);

    let data_actual_cloned = tensor_1.clone().equal(tensor_2.clone());
    let data_actual_inplace = tensor_1.equal(tensor_2);

    let data_expected = TensorData::from([[true, true, false], [true, false, false]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}

#[test]
fn test_not_equal() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 1.0], [3.0, 5.0, 4.0]]);

    let data_actual_cloned = tensor_1.clone().not_equal(tensor_2.clone());
    let data_actual_inplace = tensor_1.not_equal(tensor_2);

    let data_expected = TensorData::from([[false, false, true], [false, true, true]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}

#[test]
fn test_equal_elem() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 2.0, 5.0]]);

    let data_actual_cloned = tensor.clone().equal_elem(2);
    let data_actual_inplace = tensor.equal_elem(2);

    let data_expected = TensorData::from([[false, false, true], [false, true, false]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}

#[test]
fn test_not_equal_elem() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 2.0, 5.0]]);

    let data_actual_cloned = tensor.clone().not_equal_elem(2);
    let data_actual_inplace = tensor.not_equal_elem(2);

    let data_expected = TensorData::from([[true, true, false], [true, false, true]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}

#[test]
fn test_greater_elem() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let data_actual_cloned = tensor.clone().greater_elem(4);
    let data_actual_inplace = tensor.greater_elem(4);

    let data_expected = TensorData::from([[false, false, false], [false, false, true]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}

#[test]
fn test_greater_equal_elem() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let data_actual_cloned = tensor.clone().greater_equal_elem(4.0);
    let data_actual_inplace = tensor.greater_equal_elem(4.0);

    let data_expected = TensorData::from([[false, false, false], [false, true, true]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}

#[test]
fn test_greater() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 1.0], [3.0, 5.0, 4.0]]);

    let data_actual_cloned = tensor_1.clone().greater(tensor_2.clone());
    let data_actual_inplace = tensor_1.greater(tensor_2);

    let data_expected = TensorData::from([[false, false, true], [false, false, true]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}

#[test]
fn test_greater_equal() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 1.0], [3.0, 4.0, 5.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 5.0, 4.0]]);

    let data_actual_cloned = tensor_1.clone().greater_equal(tensor_2.clone());
    let data_actual_inplace = tensor_1.greater_equal(tensor_2);

    let data_expected = TensorData::from([[true, true, false], [true, false, true]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}

#[test]
fn test_lower_elem() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let data_actual_cloned = tensor.clone().lower_elem(4.0);
    let data_actual_inplace = tensor.lower_elem(4.0);

    let data_expected = TensorData::from([[true, true, true], [true, false, false]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}

#[test]
fn test_lower_equal_elem() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let data_actual_cloned = tensor.clone().lower_equal_elem(4.0);
    let data_actual_inplace = tensor.lower_equal_elem(4.0);

    let data_expected = TensorData::from([[true, true, true], [true, true, false]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}

#[test]
fn test_lower() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 1.0], [3.0, 4.0, 5.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 5.0, 4.0]]);

    let data_actual_cloned = tensor_1.clone().lower(tensor_2.clone());
    let data_actual_inplace = tensor_1.lower(tensor_2);

    let data_expected = TensorData::from([[false, false, true], [false, true, false]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}

#[test]
fn test_lower_equal() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 1.0], [3.0, 5.0, 4.0]]);

    let data_actual_cloned = tensor_1.clone().lower_equal(tensor_2.clone());
    let data_actual_inplace = tensor_1.lower_equal(tensor_2);

    let data_expected = TensorData::from([[true, true, false], [true, true, false]]);
    assert_eq!(data_expected, data_actual_cloned.into_data());
    assert_eq!(data_expected, data_actual_inplace.into_data());
}
