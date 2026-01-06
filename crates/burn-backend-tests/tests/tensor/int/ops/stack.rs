use super::*;
use alloc::vec;
use burn_tensor::{Tensor, TensorData};

#[test]
fn should_support_stack_ops_int() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_data([[1, 2, 3]], &device);
    let tensor_2 = TestTensorInt::<2>::from_data([[4, 5, 6]], &device);

    let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 0);
    let expected = TensorData::from([[[1, 2, 3]], [[4, 5, 6]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_generate_row_major_layout() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(1..25, &device).reshape([4, 6]);
    let zeros = TestTensorInt::zeros([4, 6], &device);
    let intersperse =
        Tensor::stack::<3>([tensor.clone(), zeros.clone()].to_vec(), 2).reshape([4, 12]);

    let expected = TensorData::from([
        [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0],
        [7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0],
        [13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0],
        [19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0],
    ]);

    intersperse.into_data().assert_eq(&expected, false);
}
