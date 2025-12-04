use crate::qtensor::*;
use crate::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn test_max_dim_2d() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.max_dim(1);
    let expected = TensorData::from([[2.], [5.]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn test_max_dim_with_indices_2d_with_dim_0th() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let (output, index) = tensor.max_dim_with_indices(0);

    let output_expected = TensorData::from([[3., 4., 5.]]);
    let index_expected = TensorData::from([[1, 1, 1]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_expected, Tolerance::rel_abs(2e-2, 1e-2));
    index.into_data().assert_eq(&index_expected, false);
}

#[test]
fn test_max_dim_with_indices_2d() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let (output, index) = tensor.max_dim_with_indices(1);

    let output_expected = TensorData::from([[2.], [5.]]);
    let index_expected = TensorData::from([[2], [2]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_expected, Tolerance::rel_abs(2e-2, 1e-2));
    index.into_data().assert_eq(&index_expected, false);
}

#[test]
fn test_min_dim_2d() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.min_dim(1);

    let expected = TensorData::from([[0.], [3.]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn test_min_dim_with_indices_2d() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let (output, index) = tensor.min_dim_with_indices(1);

    let output_expected = TensorData::from([[0.], [3.]]);
    let index_expected = TensorData::from([[0], [0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_expected, Tolerance::rel_abs(2e-2, 1e-2));
    index.into_data().assert_eq(&index_expected, false);
}

#[test]
fn test_min_dim_2d_with_0th_dim() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.min_dim(0);
    let expected = TensorData::from([[0., 1., 2.]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn test_max_dim_2d_with_0th_dim() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.max_dim(0);
    let expected = TensorData::from([[3., 4., 5.]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn test_min_dim_with_indices_2d_with_0th_dim() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let (output, index) = tensor.min_dim_with_indices(0);

    let output_expected = TensorData::from([[0., 1., 2.]]);
    let index_expected = TensorData::from([[0, 0, 0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_expected, Tolerance::rel_abs(2e-2, 1e-2));
    index.into_data().assert_eq(&index_expected, false);
}

#[test]
fn test_maximum_pair() {
    let a = QTensor::<TestBackend, 1>::int8([1.0, 5.0, 3.0, 4.0]);
    let b = QTensor::<TestBackend, 1>::int8([2.0, 1.0, 4.0, 5.0]);

    let output = a.max_pair(b);
    let expected = TensorData::from([2.0, 5.0, 4.0, 5.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn test_minimum_pair() {
    let a = QTensor::<TestBackend, 1>::int8([1.0, 5.0, 3.0, 4.0]);
    let b = QTensor::<TestBackend, 1>::int8([2.0, 1.0, 4.0, 5.0]);

    let output = a.min_pair(b);
    let expected = TensorData::from([1.0, 1.0, 3.0, 4.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}
