use crate::qtensor::*;
use crate::*;
use burn_tensor::TensorData;

#[test]
fn test_any() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([true]);
    assert_eq!(data_expected, data_actual);

    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([false]);
    assert_eq!(data_expected, data_actual);
}

#[test]
fn test_any_dim() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]]);

    let data_actual = tensor.any_dim(1).into_data();
    let data_expected = TensorData::from([[false], [true]]);
    assert_eq!(data_expected, data_actual);
}
