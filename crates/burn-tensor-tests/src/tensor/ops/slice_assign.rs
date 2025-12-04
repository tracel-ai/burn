use super::*;
use burn_tensor::{TensorData, s};

#[test]
fn should_support_slice_assign_1d() {
    let data = TensorData::from([0.0, 1.0, 2.0]);
    let data_assigned = TensorData::from([10.0, 5.0]);

    let device = Default::default();
    let tensor = TestTensor::<1>::from_data(data, &device);
    let tensor_assigned = TestTensor::<1>::from_data(data_assigned, &device);

    let output = tensor.slice_assign([0..2], tensor_assigned);
    let expected = TensorData::from([10.0, 5.0, 2.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_slice_assign_2d() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data_assigned = TensorData::from([[10.0, 5.0]]);

    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(data, &device);
    let tensor_assigned = TestTensor::<2>::from_data(data_assigned, &device);

    let output = tensor.slice_assign([1..2, 0..2], tensor_assigned);
    let expected = TensorData::from([[0.0, 1.0, 2.0], [10.0, 5.0, 5.0]]);

    output.into_data().assert_eq(&expected, false);
}

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
fn slice_assign_now_supports_non_unit_step() {
    let device = Default::default();
    // Create tensors where the shapes match for stepped slicing
    let tensor = TestTensor::<2>::ones([4, 4], &device);
    // With step=2 on first dim, we select indices 0 and 2, so we need a [2, 4] values tensor
    let values = TestTensor::<2>::zeros([2, 4], &device);

    // This now works because slice_assign supports steps != 1
    // We use s! macro to create a slice with step=2
    let result = tensor.slice_assign(s![0..3;2, ..], values);

    // Verify the result: rows 0 and 2 should be zeros, rows 1 and 3 should be ones
    let expected = TensorData::from([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]);
    result.into_data().assert_eq(&expected, false);
}

#[test]
fn test_slice_assign_with_positive_step_1d() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device);
    let values = TestTensor::<1>::from_data([10.0, 20.0, 30.0], &device);

    // Assign to indices 0, 2, 4 (step=2)
    let output = tensor.slice_assign([s![0..6;2]], values);
    let expected = TensorData::from([10.0, 2.0, 20.0, 4.0, 30.0, 6.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_slice_assign_with_positive_step_2d() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        &device,
    );

    // Assign to rows 0, 2 (step=2)
    let values = TestTensor::<2>::from_data(
        [[100.0, 101.0, 102.0, 103.0], [200.0, 201.0, 202.0, 203.0]],
        &device,
    );
    let output = tensor.clone().slice_assign([s![0..4;2]], values);
    let expected = TensorData::from([
        [100.0, 101.0, 102.0, 103.0],
        [5.0, 6.0, 7.0, 8.0],
        [200.0, 201.0, 202.0, 203.0],
        [13.0, 14.0, 15.0, 16.0],
    ]);
    output.into_data().assert_eq(&expected, false);

    // Assign to columns 0, 2 (step=2)
    let values = TestTensor::<2>::from_data(
        [
            [100.0, 200.0],
            [101.0, 201.0],
            [102.0, 202.0],
            [103.0, 203.0],
        ],
        &device,
    );
    let output = tensor.clone().slice_assign(s![.., 0..4;2], values);
    let expected = TensorData::from([
        [100.0, 2.0, 200.0, 4.0],
        [101.0, 6.0, 201.0, 8.0],
        [102.0, 10.0, 202.0, 12.0],
        [103.0, 14.0, 203.0, 16.0],
    ]);
    output.into_data().assert_eq(&expected, false);

    // Assign with step=2 on both dimensions
    let values = TestTensor::<2>::from_data([[100.0, 200.0], [300.0, 400.0]], &device);
    let output = tensor.slice_assign(s![0..4;2, 0..4;2], values);
    let expected = TensorData::from([
        [100.0, 2.0, 200.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [300.0, 10.0, 400.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_slice_assign_with_negative_step_1d() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device);
    let values = TestTensor::<1>::from_data([60.0, 50.0, 40.0, 30.0, 20.0, 10.0], &device);

    // Assign in reverse order (step=-1)
    let output = tensor.slice_assign([s![0..6;-1]], values);
    let expected = TensorData::from([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_slice_assign_with_negative_step_2d() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        &device,
    );

    // Assign to rows in reverse order (step=-1)
    let values = TestTensor::<2>::from_data(
        [
            [30.0, 31.0, 32.0, 33.0],
            [20.0, 21.0, 22.0, 23.0],
            [10.0, 11.0, 12.0, 13.0],
        ],
        &device,
    );
    let output = tensor.clone().slice_assign([s![0..3;-1]], values);
    let expected = TensorData::from([
        [10.0, 11.0, 12.0, 13.0],
        [20.0, 21.0, 22.0, 23.0],
        [30.0, 31.0, 32.0, 33.0],
    ]);
    output.into_data().assert_eq(&expected, false);

    // Assign to columns in reverse order (step=-1)
    let values = TestTensor::<2>::from_data(
        [
            [40.0, 30.0, 20.0, 10.0],
            [80.0, 70.0, 60.0, 50.0],
            [120.0, 110.0, 100.0, 90.0],
        ],
        &device,
    );
    let output = tensor.clone().slice_assign(s![.., 0..4;-1], values);
    let expected = TensorData::from([
        [10.0, 20.0, 30.0, 40.0],
        [50.0, 60.0, 70.0, 80.0],
        [90.0, 100.0, 110.0, 120.0],
    ]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_slice_assign_with_mixed_steps() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        &device,
    );

    // Positive step along rows, negative along columns
    let values = TestTensor::<2>::from_data(
        [[100.0, 101.0, 102.0, 103.0], [200.0, 201.0, 202.0, 203.0]],
        &device,
    );
    let output = tensor.clone().slice_assign(s![0..4;2, 0..4;-1], values);
    let expected = TensorData::from([
        [103.0, 102.0, 101.0, 100.0],
        [5.0, 6.0, 7.0, 8.0],
        [203.0, 202.0, 201.0, 200.0],
        [13.0, 14.0, 15.0, 16.0],
    ]);
    output.into_data().assert_eq(&expected, false);

    // Negative step along rows, positive along columns
    let values = TestTensor::<2>::from_data(
        [
            [100.0, 200.0],
            [101.0, 201.0],
            [102.0, 202.0],
            [103.0, 203.0],
        ],
        &device,
    );
    let output = tensor.slice_assign(s![0..4;-1, 0..4;2], values);
    let expected = TensorData::from([
        [103.0, 2.0, 203.0, 4.0],
        [102.0, 6.0, 202.0, 8.0],
        [101.0, 10.0, 201.0, 12.0],
        [100.0, 14.0, 200.0, 16.0],
    ]);
    output.into_data().assert_eq(&expected, false);
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
fn test_slice_assign_3d_with_steps() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        &device,
    );

    // Test step=2 along first dimension
    let values = TestTensor::<3>::from_data(
        [
            [[100.0, 101.0], [102.0, 103.0]],
            [[200.0, 201.0], [202.0, 203.0]],
        ],
        &device,
    );
    let output = tensor.clone().slice_assign(s![0..4;2, .., ..], values);
    let expected = TensorData::from([
        [[100.0, 101.0], [102.0, 103.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[200.0, 201.0], [202.0, 203.0]],
        [[13.0, 14.0], [15.0, 16.0]],
    ]);
    output.into_data().assert_eq(&expected, false);

    // Test step=-1 along all dimensions
    let values = TestTensor::<3>::from_data(
        [
            [[400.0, 399.0], [398.0, 397.0]],
            [[396.0, 395.0], [394.0, 393.0]],
            [[392.0, 391.0], [390.0, 389.0]],
            [[388.0, 387.0], [386.0, 385.0]],
        ],
        &device,
    );
    let output = tensor.slice_assign(s![0..4;-1, 0..2;-1, 0..2;-1], values);
    let expected = TensorData::from([
        [[385.0, 386.0], [387.0, 388.0]],
        [[389.0, 390.0], [391.0, 392.0]],
        [[393.0, 394.0], [395.0, 396.0]],
        [[397.0, 398.0], [399.0, 400.0]],
    ]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_slice_assign_partial_with_steps() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ],
        &device,
    );

    // Assign to a subset with step=2
    let values = TestTensor::<2>::from_data([[100.0, 200.0], [300.0, 400.0]], &device);
    let output = tensor.slice_assign(s![1..4;2, 1..4;2], values);
    let expected = TensorData::from([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 100.0, 8.0, 200.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0],
        [16.0, 300.0, 18.0, 400.0, 20.0],
        [21.0, 22.0, 23.0, 24.0, 25.0],
    ]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_slice_assign_empty_range() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let values: TestTensor<2> = TestTensor::empty([2, 0], &device);

    // Empty slice assignment (start == end) should be a no-op
    let output = tensor.clone().slice_assign([0..2, 1..1], values);
    let expected = TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_slice_assign_empty_range_1d() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0, 4.0, 5.0], &device);
    let values: TestTensor<1> = TestTensor::empty([0], &device);

    // Empty slice assignment should return tensor unchanged
    let output = tensor.clone().slice_assign([2..2], values);
    let expected = TensorData::from([1.0, 2.0, 3.0, 4.0, 5.0]);

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
