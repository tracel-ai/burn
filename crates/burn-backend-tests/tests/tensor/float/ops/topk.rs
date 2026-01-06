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
