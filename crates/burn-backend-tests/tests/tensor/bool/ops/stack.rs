use super::*;
use alloc::vec;
use burn_tensor::{Tensor, TensorData};

#[test]
fn should_support_stack_ops_bool() {
    let device = Default::default();
    let tensor_1 = TestTensorBool::<2>::from_data([[false, true, true]], &device);
    let tensor_2 = TestTensorBool::<2>::from_data([[true, true, false]], &device);

    let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 0);
    let expected = TensorData::from([[[false, true, true]], [[true, true, false]]]);

    output.into_data().assert_eq(&expected, false);
}
