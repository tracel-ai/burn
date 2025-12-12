use super::qtensor::*;
use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn test_topk_1d() {
    let tensor = QTensor::<TestBackend, 1>::int8([1.0, 2.0, 3.0, 4.0, 5.0]);

    let values = tensor.topk(3, /*dim*/ 0);
    let expected = TensorData::from([5., 4., 3.]);

    values
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn test_topk() {
    let tensor = QTensor::<TestBackend, 3>::int8([
        [[1., 4., 7.], [2., 5., 6.]],
        [[3., 0., 9.], [8., 2., 7.]],
    ]);

    let values = tensor.topk(2, /*dim*/ 2);
    let expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 7.]]]);

    values
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn test_topk_with_indices() {
    // 1D
    let tensor = QTensor::<TestBackend, 1>::int8([1.0, 2.0, 3.0, 4.0, 5.0]);

    let (values, indices) = tensor.topk_with_indices(3, /*dim*/ 0);

    let values_expected = TensorData::from([5., 4., 3.]);
    values
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::permissive());

    let indices_expected = TensorData::from([4, 3, 2]);
    indices.into_data().assert_eq(&indices_expected, false);
}

#[test]
fn test_topk_with_indices_3d() {
    // 3D
    let tensor = QTensor::<TestBackend, 3>::int8([
        [[1., 4., 7.], [2., 5., 6.]],
        [[3., 0., 9.], [8., 2., 7.]],
    ]);

    let (values, indices) = tensor.topk_with_indices(2, /*dim*/ 2);

    let values_expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 7.]]]);

    values
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::absolute(1e-1));

    let indices_expected = TensorData::from([[[2, 1], [2, 1]], [[2, 0], [0, 2]]]);

    indices.into_data().assert_eq(&indices_expected, false);
}
