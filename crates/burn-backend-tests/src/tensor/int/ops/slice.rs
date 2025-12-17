use super::*;
use burn_tensor::{TensorData, s};

#[test]
fn slice_should_not_corrupt_potentially_inplace_operations() {
    let tensor = TestTensorInt::<1>::from([1, 2, 3, 4, 5]);
    let tensor = tensor.clone().slice([0..3]) + tensor.clone().slice([2..5]);

    let expected = TensorData::from([4, 6, 8]);

    tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn test_slice_int_tensor_with_steps() {
    let device = Default::default();
    let tensor =
        TestTensorInt::<2>::from_data([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], &device);

    // Test step=2 along first dimension
    let sliced = tensor.clone().slice([s![0..3;2]]);
    let expected = TensorData::from([[1i32, 2, 3, 4], [9, 10, 11, 12]]);
    sliced.into_data().assert_eq(&expected, false);

    // Test step=-1 along second dimension
    let sliced = tensor.clone().slice(s![.., 0..4;-1]);
    let expected = TensorData::from([[4i32, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]]);
    sliced.into_data().assert_eq(&expected, false);
}
