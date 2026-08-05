use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn test_topk_with_indices_3d() {
    let tensor =
        TestTensor::<3>::from([[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 7.]]]);

    let (values, indices) = tensor.topk_with_indices(2, /*dim*/ 2);

    let values_expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 7.]]]);

    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());

    let indices_expected = TensorData::from([[[2, 1], [2, 1]], [[2, 0], [0, 2]]]);

    indices.into_data().assert_eq(&indices_expected, false);
}

#[test]
fn test_topk_supports_negative_dim_float() {
    let tensor =
        TestTensor::<3>::from([[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 7.]]]);

    let values = tensor.topk(2, -1);
    let values_expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 7.]]]);

    values
        .into_data()
        .assert_approx_eq::<FloatElem>(&values_expected, Tolerance::default());
}

#[test]
fn test_topk_supports_k_dim_size() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[12., -2., 3.], [5., 3., 6.]]),
        &Default::default(),
    );

    let values = tensor.clone().topk(2, 0);
    values.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[12., 3., 6.], [5., -2., 3.]]),
        Tolerance::default(),
    );

    let values = tensor.topk(3, 1);
    values.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[12., 3., -2.], [6., 5., 3.]]),
        Tolerance::default(),
    );

    let tensor =
        TestTensor::<3>::from([[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 7.]]]);

    let values = tensor.topk(2, 0);
    values.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[[3., 4., 9.], [8., 5., 7.]], [[1., 0., 7.], [2., 2., 6.]]]),
        Tolerance::default(),
    );
}

#[test]
#[should_panic]
fn test_topk_should_panic_k_larger() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[12., -2., 3.], [5., 3., 6.]]),
        &Default::default(),
    );

    // k=3 is too large for dim of size 2
    let _values = tensor.topk(3, 0);
}

#[test]
#[should_panic]
fn test_topk_with_indices_should_panic_k_larger() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[12., -2., 3.], [5., 3., 6.]]),
        &Default::default(),
    );

    // k=3 is too large for dim of size 2
    let (_values, _indices) = tensor.topk_with_indices(3, 0);
}
