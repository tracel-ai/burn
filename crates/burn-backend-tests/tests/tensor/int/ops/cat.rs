use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_cat_ops_int() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_data([[1, 2, 3]], &device);
    let tensor_2 = TestTensorInt::<2>::from_data([[4, 5, 6]], &device);

    let output = Tensor::cat(vec![tensor_1, tensor_2], 0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 2, 3], [4, 5, 6]]), false);
}

#[test]
fn should_support_cat_with_empty_tensor_int() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_data([[1, 2, 3]], &device);
    let tensor_2: TestTensorInt<2> = TestTensorInt::empty([1, 0], &device);

    let output = Tensor::cat(vec![tensor_1, tensor_2], 1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 2, 3]]), false);
}
