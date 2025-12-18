use super::*;
use burn_tensor::{DType, TensorData};

#[test]
fn cast_int_to_bool() {
    let tensor1 = TestTensorInt::<2>::from([[0, 43, 0], [2, -4, 31]]);
    let data_actual = tensor1.bool().into_data();
    let data_expected = TensorData::from([[false, true, false], [true, true, true]]);
    data_actual.assert_eq(&data_expected, false);
}

#[test]
fn cast_bool_to_int_tensor() {
    let tensor = TestTensorBool::<2>::from([[true, false, true], [false, false, true]]).int();

    let expected = TensorData::from([[1, 0, 1], [0, 0, 1]]);

    tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn cast_int_precision() {
    let data = TensorData::from([[1, 2, 3], [4, 5, 6]]);
    let tensor = TestTensorInt::<2>::from(data.clone());

    let output = tensor.cast(DType::I32);

    assert_eq!(output.dtype(), DType::I32);
    output.into_data().assert_eq(&data, false);
}
