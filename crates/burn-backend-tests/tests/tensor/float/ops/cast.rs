use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{DType, FloatDType, IntDType, TensorData};

#[test]
fn cast_float_to_bool() {
    let tensor1 = TestTensor::<2>::from([[0.0, 43.0, 0.0], [2.0, -4.2, 31.33]]);
    let data_actual = tensor1.bool().into_data();
    let data_expected = TensorData::from([[false, true, false], [true, true, true]]);
    data_actual.assert_eq(&data_expected, false);
}

#[test]
fn cast_float_to_int() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.4, 5.5, 6.6]]).int();
    let expected = TensorData::from([[1, 2, 3], [4, 5, 6]]);

    tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn cast_int_to_float_tensor() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]).float();

    let expected = TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn cast_bool_to_float_tensor() {
    let tensor = TestTensorBool::<2>::from([[true, false, true], [false, false, true]]).float();

    let expected = TensorData::from([[1., 0., 1.], [0., 0., 1.]]);

    tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn cast_float_precision() {
    let data = TensorData::from([[1.0, 2.0, 3.0], [4.4, 5.5, 6.6]]);
    let tensor = TestTensor::<2>::from(data.clone());

    let output = tensor.cast(DType::F32);

    assert_eq!(output.dtype(), DType::F32);
    // Use precision 2 for parameterized tests in f16 and bf16
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&data, Tolerance::default());
}

#[test]
fn cast_float_to_int_with_dtype() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.4, 5.5, 6.6]]);

    let output = tensor.cast(IntDType::I32);

    assert_eq!(output.dtype(), DType::I32);
    let expected = TensorData::from([[1i32, 2, 3], [4, 5, 6]]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn cast_float_within_kind_with_float_dtype() {
    let tensor = TestTensor::<1>::from([1.0, 2.5, 3.7]);

    let output = tensor.cast(FloatDType::F32);

    assert_eq!(output.dtype(), DType::F32);
}

#[test]
fn cast_float_same_dtype_is_noop() {
    let data = TensorData::from([1.0, 2.0, 3.0]);
    let tensor = TestTensor::<1>::from(data.clone());
    let original_dtype = tensor.dtype();
    let float_dtype: FloatDType = original_dtype.into();

    let output = tensor.cast(float_dtype);

    assert_eq!(output.dtype(), original_dtype);
    output.into_data().assert_eq(&data, false);
}

#[test]
#[should_panic]
fn cast_float_with_int_dtype_panics() {
    let tensor = TestTensor::<1>::from([1.0, 2.0]);
    let _ = tensor.cast(DType::I32);
}
