use super::*;
use burn_tensor::{DType, FloatDType, IntDType, TensorData};

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

#[test]
fn cast_int_to_float_with_dtype() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);

    let output = tensor.cast(FloatDType::F32);

    assert_eq!(output.dtype(), DType::F32);
    let expected = TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn cast_int_within_kind() {
    let tensor = TestTensorInt::<1>::from([1, 2, 3]);

    let output = tensor.cast(IntDType::I32);

    assert_eq!(output.dtype(), DType::I32);
    let expected = TensorData::from([1i32, 2, 3]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn cast_int_same_dtype_is_noop() {
    let data = TensorData::from([1, 2, 3]);
    let tensor = TestTensorInt::<1>::from(data.clone());
    let original_dtype = tensor.dtype();
    let int_dtype: IntDType = original_dtype.into();

    let output = tensor.cast(int_dtype);

    assert_eq!(output.dtype(), original_dtype);
    output.into_data().assert_eq(&data, false);
}

#[test]
#[should_panic]
fn cast_int_with_float_dtype_panics() {
    let tensor = TestTensorInt::<1>::from([1, 2]);
    let _ = tensor.cast(DType::F32);
}

#[test]
fn test_int_into_float_flipped() {
    // [1, 2, 3, 4] flipped -> [4, 3, 2, 1] -> float [4.0, 3.0, 2.0, 1.0]
    let t = TestTensorInt::<1>::from([1, 2, 3, 4]).flip([0]);

    let output = t.float();

    output
        .into_data()
        .assert_eq(&TensorData::from([4.0f32, 3.0, 2.0, 1.0]), false);
}
