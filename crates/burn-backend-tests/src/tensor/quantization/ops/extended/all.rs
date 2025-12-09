use super::qtensor::*;
use super::*;
use burn_tensor::TensorData;

#[test]
fn test_all() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]);
    let data_actual = tensor.all().into_data();
    let data_expected = TensorData::from([false]);
    assert_eq!(data_expected, data_actual);

    let tensor = QTensor::<TestBackend, 2>::int8([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    let data_actual = tensor.all().into_data();
    let data_expected = TensorData::from([true]);
    assert_eq!(data_expected, data_actual);
}

#[test]
fn test_all_dim() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]);
    let data_actual = tensor.all_dim(1).into_data();
    let data_expected = TensorData::from([[false], [true]]);
    assert_eq!(data_expected, data_actual);
}
