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
