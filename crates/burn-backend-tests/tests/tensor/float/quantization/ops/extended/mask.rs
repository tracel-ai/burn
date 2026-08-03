use super::qtensor::*;
use super::*;
use burn_tensor::DType;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_mask_where_ops() {
    let tensor = QTensor::<2>::int8([[1.0, 7.0], [2.0, 3.0]]);
    let mask = TestTensorBool::<2>::from_bool(
        TensorData::from([[true, false], [false, true]]),
        &Default::default(),
    );
    let value = QTensor::<2>::int8([[1.8, 2.8], [3.8, 4.8]]);

    let output = tensor.mask_where(mask, value);
    let expected = TensorData::from([[1.8, 7.0], [2.0, 4.8]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_mask_fill_ops() {
    let tensor = QTensor::<2>::int8([[1.0, 7.0], [2.0, 3.0]]);
    let mask = TestTensorBool::<2>::from_bool(
        TensorData::from([[true, false], [false, true]]),
        &Default::default(),
    );

    let output = tensor.mask_fill(mask, 2.0);
    let expected = TensorData::from([[2.0, 7.0], [2.0, 2.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_mask_select_ops() {
    let tensor = QTensor::<2>::int8([[1.0, 7.0], [2.0, 3.0]]);
    let mask = TestTensorBool::<2>::from_bool(
        TensorData::from([[true, false], [false, true]]),
        &Default::default(),
    );

    let output = tensor.mask_select(mask);
    let expected = TensorData::from([1.0, 3.0]);

    // `mask_select` only moves existing elements around, so it stays quantized instead of
    // dequantizing like `mask_where` and `mask_fill` do.
    assert!(matches!(output.to_data().dtype, DType::QFloat(_)));

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_mask_select_ops_empty_mask() {
    let tensor = QTensor::<2>::int8([[1.0, 7.0], [2.0, 3.0]]);
    let mask = TestTensorBool::<2>::from_bool(
        TensorData::from([[false, false], [false, false]]),
        &Default::default(),
    );

    let output = tensor.mask_select(mask);

    // Selecting nothing still has to keep the quantized kind, not fall back to a float tensor.
    assert_eq!(output.dims(), [0]);
    assert!(matches!(output.to_data().dtype, DType::QFloat(_)));
}
