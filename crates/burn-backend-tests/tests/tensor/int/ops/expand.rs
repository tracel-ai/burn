use super::*;
use burn_tensor::TensorData;

#[test]
fn expand_2d_int() {
    let tensor = TestTensorInt::<1>::from([1, 2, 3]);
    let output = tensor.expand([3, 3]);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), false);
}

#[test]
fn should_all_negative_one() {
    let tensor = TestTensorInt::<1>::from([1, 2, 3]);
    let output = tensor.expand([2, -1]);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 2, 3], [1, 2, 3]]), false);
}

#[test]
#[should_panic]
fn should_panic_negative_one_on_non_existing_dim() {
    let tensor = TestTensorInt::<1>::from([1, 2, 3]);
    let _expanded_tensor = tensor.expand([-1, 3]);
}

/// Regression test for https://github.com/tracel-ai/burn/issues/2091
#[test]
fn inplace_op_after_expand() {
    let tensor = TestTensorInt::<1>::from([1, 2, 3]);
    let mut output = tensor.expand([2, 3]);
    output = output + 1;

    output
        .into_data()
        .assert_eq(&TensorData::from([[2, 3, 4], [2, 3, 4]]), false);
}
