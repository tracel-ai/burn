use super::*;
use burn_tensor::{DType, FloatDType, IntDType, TensorData};

#[test]
fn cast_bool_to_int_with_dtype() {
    let tensor = TestTensorBool::<2>::from([[true, false, true], [false, false, true]]);

    let output = tensor.cast(IntDType::I32);

    assert_eq!(output.dtype(), DType::I32);
    let expected = TensorData::from([[1i32, 0, 1], [0, 0, 1]]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn cast_bool_to_float_with_dtype() {
    let tensor = TestTensorBool::<2>::from([[true, false, true], [false, false, true]]);

    let output = tensor.cast(FloatDType::F32);

    assert_eq!(output.dtype(), DType::F32);
    let expected = TensorData::from([[1.0f32, 0.0, 1.0], [0.0, 0.0, 1.0]]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_bool_into_int_flipped() {
    // [T, F, T, F] flipped -> [F, T, F, T] -> int [0, 1, 0, 1]
    let t = TestTensorBool::<1>::from([true, false, true, false]).flip([0]);

    let output = t.int();

    output
        .into_data()
        .assert_eq(&TensorData::from([0i64, 1, 0, 1]), false);
}

#[test]
fn test_bool_into_float_flipped() {
    let t = TestTensorBool::<1>::from([true, false, true, false]).flip([0]);

    let output = t.float();

    output
        .into_data()
        .assert_eq(&TensorData::from([0.0f32, 1.0, 0.0, 1.0]), false);
}

#[test]
fn test_bool_into_int_flipped_2d() {
    // [[T, F], [F, T]] flipped on axis 0 -> [[F, T], [T, F]] -> int [[0, 1], [1, 0]]
    let t = TestTensorBool::<2>::from([[true, false], [false, true]]).flip([0]);

    let output = t.int();

    output
        .into_data()
        .assert_eq(&TensorData::from([[0i64, 1], [1, 0]]), false);
}
