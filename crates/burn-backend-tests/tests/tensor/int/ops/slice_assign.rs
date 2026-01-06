use super::*;
use burn_tensor::{TensorData, s};

#[test]
fn slice_assign_should_not_corrupt_potentially_inplace_operations() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::from_data([1, 2, 3, 4, 5], &device);
    let values = TestTensorInt::<1>::from_data([10, 20, 30], &device);
    let tensor_1 = tensor.clone().slice_assign([0..3], values);
    let tensor_2 = tensor + 2;

    let expected = TensorData::from([10, 20, 30, 4, 5]);

    tensor_1.into_data().assert_eq(&expected, false);

    let expected = TensorData::from([3, 4, 5, 6, 7]);

    tensor_2.into_data().assert_eq(&expected, false);
}

#[test]
fn test_slice_assign_int_tensor_with_steps() {
    let device = Default::default();
    let tensor =
        TestTensorInt::<2>::from_data([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], &device);

    // Test step=2 along first dimension
    let values =
        TestTensorInt::<2>::from_data([[100, 101, 102, 103], [200, 201, 202, 203]], &device);
    let output = tensor.clone().slice_assign([s![0..3;2]], values);
    let expected = TensorData::from([[100i32, 101, 102, 103], [5, 6, 7, 8], [200, 201, 202, 203]]);
    output.into_data().assert_eq(&expected, false);

    // Test step=-1 along second dimension
    let values = TestTensorInt::<2>::from_data(
        [[40, 30, 20, 10], [80, 70, 60, 50], [120, 110, 100, 90]],
        &device,
    );
    let output = tensor.slice_assign(s![.., 0..4;-1], values);
    let expected = TensorData::from([[10i32, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_slice_assign_empty_range_int() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::from_data([1, 2, 3, 4, 5], &device);
    let values: TestTensorInt<1> = TestTensorInt::empty([0], &device);

    // Empty slice assignment for int tensor
    let output = tensor.clone().slice_assign([3..3], values);
    let expected = TensorData::from([1i32, 2, 3, 4, 5]);

    output.into_data().assert_eq(&expected, false);
}
