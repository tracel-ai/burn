use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_powi_broadcast() {
    // [1, 3] broadcast against [2, 3]
    let tensor_1 = TestTensorInt::<2>::from([[1, 2, 3]]);
    let tensor_2 = TestTensorInt::from([[2, 2, 2], [3, 3, 3]]);

    let output = tensor_1.clone().powi(tensor_2);
    let expected = TensorData::from([[1, 4, 9], [1, 8, 27]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic(expected = "The provided tensors have incompatible shapes.")]
fn should_panic_powi_incompatible_shapes() {
    // Same rank, but [2, 2] vs [2, 3]: dimension 1 cannot broadcast.
    let tensor_1 = TestTensorInt::<2>::from([[1, 2], [3, 4]]);
    let tensor_2 = TestTensorInt::from([[2, 2, 2], [3, 3, 3]]);

    let output = tensor_1.powi(tensor_2);
    output.into_data();
}
