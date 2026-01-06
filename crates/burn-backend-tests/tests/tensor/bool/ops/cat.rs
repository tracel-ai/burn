use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_cat_ops_bool() {
    let device = Default::default();
    let tensor_1 = TestTensorBool::<2>::from_data([[false, true, true]], &device);
    let tensor_2 = TestTensorBool::<2>::from_data([[true, true, false]], &device);

    let output = Tensor::cat(vec![tensor_1, tensor_2], 0);

    output.into_data().assert_eq(
        &TensorData::from([[false, true, true], [true, true, false]]),
        false,
    );
}

#[test]
fn should_support_cat_with_empty_tensor_bool() {
    let device = Default::default();
    let tensor_1 = TestTensorBool::<2>::from_data([[true, false, true]], &device);
    let tensor_2: TestTensorBool<2> = TestTensorBool::empty([1, 0], &device);

    let output = Tensor::cat(vec![tensor_1, tensor_2], 1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[true, false, true]]), false);
}
