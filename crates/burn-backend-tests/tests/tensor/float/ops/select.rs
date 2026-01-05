use super::*;
use burn_tensor::{IndexingUpdateOp, TensorData};

#[test]
fn should_select_1d() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([0.0, 1.0, 2.0], &device);
    let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &device);

    let output = tensor.select(0, indices);
    let expected = TensorData::from([1.0, 1.0, 0.0, 1.0, 2.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_2d_dim0_same_num_dim() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
    let indices = TestTensorInt::from_data([1, 0], &device);

    let output = tensor.select(0, indices);
    let expected = TensorData::from([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_2d_dim0_more_num_dim() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
    let indices = TestTensorInt::from_data([1, 0, 1, 1], &device);

    let output = tensor.select(0, indices);
    let expected = TensorData::from([
        [3.0, 4.0, 5.0],
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_2d_dim0_vec() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], &device);
    let indices = TestTensorInt::from_data([1, 0, 3, 2], &device);

    let output = tensor.select(0, indices);
    let expected = TensorData::from([[2.0, 3.0], [0.0, 1.0], [6.0, 7.0], [4.0, 5.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_2d_dim1() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
    let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &device);

    let output = tensor.select(1, indices);
    let expected = TensorData::from([[1.0, 1.0, 0.0, 1.0, 2.0], [4.0, 4.0, 3.0, 4.0, 5.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_add_1d() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([0.0, 1.0, 2.0], &device);
    let values = TestTensor::from_data([5.0, 4.0, 3.0, 2.0, 1.0], &device);
    let indices = TestTensorInt::from_data(TensorData::from([1, 1, 0, 1, 2]), &device);

    let output = tensor.select_assign(0, indices, values, IndexingUpdateOp::Add);
    let expected = TensorData::from([3.0, 12.0, 3.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_add_1d_int() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::from_data([7, 8, 9], &device);
    let values = TestTensorInt::from_data([5, 4, 3, 2, 1], &device);
    let indices = TestTensorInt::from_data(TensorData::from([1, 1, 0, 1, 2]), &device);

    let output = tensor.select_assign(0, indices, values, IndexingUpdateOp::Add);
    let expected = TensorData::from([10, 19, 10]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_add_2d_dim0() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
    let values = TestTensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let indices = TestTensorInt::from_data(TensorData::from([1, 0]), &device);

    let output = tensor.select_assign(0, indices, values, IndexingUpdateOp::Add);
    let expected = TensorData::from([[4.0, 6.0, 8.0], [4.0, 6.0, 8.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_add_2d_dim1() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
    let values = TestTensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let indices = TestTensorInt::from_data(TensorData::from([1, 0, 2]), &device);

    let output = tensor.select_assign(1, indices, values, IndexingUpdateOp::Add);
    let expected = TensorData::from([[2.0, 2.0, 5.0], [8.0, 8.0, 11.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_3d_dim1_vec() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0]],
        ],
        &device,
    );
    let indices = TestTensorInt::from_data([1, 0, 3, 2], &device);

    let output = tensor.select(1, indices);
    let expected = TensorData::from([
        [[3.0, 4.0], [1.0, 2.0], [7.0, 8.0], [5.0, 6.0]],
        [[-3.0, -4.0], [-1.0, -2.0], [-7.0, -8.0], [-5.0, -6.0]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic]
fn should_select_panic_invalid_dimension() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
    let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &device);

    tensor.select(10, indices);
}

#[test]
fn should_match_default_implementation_behavior() {
    // Verify optimized implementation matches original default logic
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([true, false, true], &device);
    let indices = TestTensorInt::from_data([0, 1, 0], &device);
    let values = TestTensorBool::<1>::from_data([false, true, true], &device);

    let optimized_result =
        tensor
            .clone()
            .select_assign(0, indices.clone(), values.clone(), IndexingUpdateOp::Add);

    // Manual default implementation logic
    let int_tensor = tensor.int();
    let int_values = values.int();
    let assigned = int_tensor.select_assign(0, indices, int_values, IndexingUpdateOp::Add);
    let default_result = assigned.greater_elem(0);

    optimized_result
        .into_data()
        .assert_eq(&default_result.into_data(), false);
}

#[test]
fn should_select_with_negative_dim_2d() {
    // Test using negative dimension indexing on 2D tensor
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
    let indices = TestTensorInt::from_data([1, 0, 2], &device);

    // Using -1 should refer to the last dimension (dim 1)
    let output_neg = tensor.clone().select(-1, indices.clone());
    let output_pos = tensor.select(1, indices);

    // Both should produce the same result
    output_neg
        .into_data()
        .assert_eq(&output_pos.into_data(), false);
}

#[test]
fn should_select_add_with_negative_dim_2d() {
    // Test select_add with negative dimension on 2D tensor
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
    let values = TestTensor::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let indices = TestTensorInt::from_data([0, 2], &device);

    // Using -1 should refer to the last dimension (dim 1)
    let output_neg =
        tensor
            .clone()
            .select_assign(-1, indices.clone(), values.clone(), IndexingUpdateOp::Add);
    let output_pos = tensor.select_assign(1, indices, values, IndexingUpdateOp::Add);

    output_neg
        .into_data()
        .assert_eq(&output_pos.into_data(), false);
}

#[test]
#[should_panic]
fn should_panic_select_negative_dim_out_of_bounds() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let indices = TestTensorInt::from_data([0, 1], &device);

    // This should panic because -3 is out of bounds for a 2D tensor
    tensor.select(-3, indices);
}

#[test]
#[should_panic]
fn should_panic_select_add_negative_dim_out_of_bounds() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let values = TestTensor::from_data([[5.0], [6.0]], &device);
    let indices = TestTensorInt::from_data([0], &device);

    // This should panic because -3 is out of bounds for a 2D tensor
    tensor.select_assign(-3, indices, values, IndexingUpdateOp::Add);
}
