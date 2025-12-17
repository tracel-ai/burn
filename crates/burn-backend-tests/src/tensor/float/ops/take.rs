use super::*;
use burn_tensor::TensorData;

#[test]
fn should_take_1d() {
    // Test that take works with 1D indices
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([0.0, 1.0, 2.0], &device);
    let indices = TestTensorInt::<1>::from_data([1, 1, 0, 1, 2], &device);

    let output = tensor.take::<1, 1>(0, indices);
    let expected = TensorData::from([1.0, 1.0, 0.0, 1.0, 2.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_take_2d_dim0() {
    // Test take on 2D tensor along dimension 0
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
    let indices = TestTensorInt::<1>::from_data([1, 0, 1, 1], &device);

    let output = tensor.take::<1, 2>(0, indices);
    let expected = TensorData::from([
        [3.0, 4.0, 5.0],
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_take_2d_dim1() {
    // Test take on 2D tensor along dimension 1
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
    let indices = TestTensorInt::<1>::from_data([2, 0, 1], &device);

    let output = tensor.take::<1, 2>(1, indices);
    let expected = TensorData::from([[2.0, 0.0, 1.0], [5.0, 3.0, 4.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn take_and_select_should_be_equivalent() {
    // Verify that take and select produce identical results
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        &device,
    );
    let indices = TestTensorInt::<1>::from_data([2, 0, 1, 1], &device);

    let result_take = tensor.clone().take::<1, 2>(0, indices.clone());
    let result_select = tensor.select(0, indices);

    let take_data = result_take.into_data();
    let select_data = result_select.into_data();

    take_data.assert_eq(&select_data, false);
}

#[test]
fn should_take_with_2d_indices() {
    // Test take with 2D indices - output will be 3D with shape [2, 2, 4]
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        &device,
    );

    // 2D indices to select along dimension 0 - shape [2, 2]
    let indices = TestTensorInt::<2>::from_data([[0, 2], [1, 0]], &device);
    let output = tensor.take::<2, 3>(0, indices);

    // Expected: shape [2, 2, 4] - indices shape replaces dim 0
    let expected = TensorData::from([
        [[1.0, 2.0, 3.0, 4.0], [9.0, 10.0, 11.0, 12.0]],
        [[5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_take_with_2d_indices_dim1() {
    // Test take with 2D indices along dimension 1 - output will be 3D with shape [2, 2, 2]
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device);

    // 2D indices to select along dimension 1 - shape [2, 2]
    let indices = TestTensorInt::<2>::from_data([[0, 3], [2, 1]], &device);
    let output = tensor.take::<2, 3>(1, indices);

    // Expected: shape [2, 2, 2] - indices shape replaces dim 1
    let expected = TensorData::from([[[1.0, 4.0], [3.0, 2.0]], [[5.0, 8.0], [7.0, 6.0]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_take_3d_tensor() {
    // Test take with 3D tensor - output will be 4D with shape [2, 2, 2, 2]
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        &device,
    );

    // 2D indices to select along dimension 1 - shape [2, 2]
    let indices = TestTensorInt::<2>::from_data([[0, 2], [1, 0]], &device);
    let output = tensor.take::<2, 4>(1, indices);

    // Expected: shape [2, 2, 2, 2] - indices shape replaces dim 1
    let expected = TensorData::from([
        [[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [1.0, 2.0]]],
        [[[7.0, 8.0], [11.0, 12.0]], [[9.0, 10.0], [7.0, 8.0]]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_take_with_3d_indices() {
    // Test take with 3D indices - output will be 4D
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

    // 3D indices to select along dimension 1 - shape [2, 2, 2]
    let indices = TestTensorInt::<3>::from_data([[[0, 2], [1, 0]], [[2, 1], [0, 2]]], &device);
    let output = tensor.take::<3, 4>(1, indices);

    // Expected: shape [2, 2, 2, 2] - indices shape replaces dim 1
    let expected = TensorData::from([
        [[[1.0, 3.0], [2.0, 1.0]], [[3.0, 2.0], [1.0, 3.0]]],
        [[[4.0, 6.0], [5.0, 4.0]], [[6.0, 5.0], [4.0, 6.0]]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic]
fn should_panic_take_invalid_dimension() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
    let indices = TestTensorInt::<1>::from_data([1, 0], &device);

    // This should panic because dimension 10 is out of bounds
    tensor.take::<1, 2>(10, indices);
}

#[test]
fn should_take_with_single_index() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let indices = TestTensorInt::<1>::from_data([1], &device);

    let output = tensor.take::<1, 2>(0, indices);
    let expected = TensorData::from([[4.0, 5.0, 6.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_take_with_negative_dim_2d() {
    // Test using negative dimension indexing on 2D tensor
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let indices = TestTensorInt::<1>::from_data([2, 0, 1], &device);

    // Using -1 should refer to the last dimension (dim 1)
    let output_neg = tensor.clone().take::<1, 2>(-1, indices.clone());
    let output_pos = tensor.take::<1, 2>(1, indices);

    // Both should produce the same result
    let neg_data = output_neg.into_data();
    let pos_data = output_pos.into_data();
    neg_data.assert_eq(&pos_data, false);
}

#[test]
#[should_panic]
fn should_panic_take_negative_dim_out_of_bounds() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let indices = TestTensorInt::<1>::from_data([0, 1], &device);

    // This should panic because -3 is out of bounds for a 2D tensor
    tensor.take::<1, 2>(-3, indices);
}
