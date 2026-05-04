use super::*;
use burn_tensor::TensorData;

#[test]
fn test_topk_with_indices_1d() {
    let tensor = TestTensorInt::<1>::from([1, 2, 3, 4, 5]);

    let (values, indices) = tensor.topk_with_indices(3, /*dim*/ 0);

    let values_expected = TensorData::from([5, 4, 3]);
    values.into_data().assert_eq(&values_expected, false);

    let indices_expected = TensorData::from([4, 3, 2]);
    indices.into_data().assert_eq(&indices_expected, false);
}
