use super::*;
use burn_tensor::{IndexingUpdateOp, TensorData};

#[test]
fn should_select_bool_tensor_1d() {
    // Test that select works for boolean tensors
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([true, false, true], &device);
    let indices = TestTensorInt::from_data([0, 2, 1, 0], &device);

    let output = tensor.select(0, indices);
    let expected = TensorData::from([true, true, false, true]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_bool_tensor_2d() {
    // Test that select works for boolean 2D tensors
    let device = Default::default();
    let tensor =
        TestTensorBool::<2>::from_data([[true, false, true], [false, true, false]], &device);
    let indices = TestTensorInt::from_data([1, 0], &device);

    let output = tensor.select(0, indices);
    let expected = TensorData::from([[false, true, false], [true, false, true]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_add_bool_tensor() {
    // Test that select_add works for boolean tensors
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([true, false, true], &device);
    let values = TestTensorBool::<1>::from_data([false, true], &device);
    let indices = TestTensorInt::from_data([0, 2], &device);

    let output = tensor.select_assign(0, indices, values, IndexingUpdateOp::Add);
    // Note: select_add uses sum reduction, so:
    // index 0: true OR false = true
    // index 2: true OR true = true
    // index 1: false (unchanged)
    let expected = TensorData::from([true, false, true]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_add_bool_overlapping_indices() {
    // Test accumulation behavior with overlapping indices
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([false, true], &device);
    let indices = TestTensorInt::from_data([0, 0], &device);
    let values = TestTensorBool::<1>::from_data([true, false], &device);

    let output = tensor.select_assign(0, indices, values, IndexingUpdateOp::Add);
    // Index 0: false OR true OR false = true
    let expected = TensorData::from([true, true]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_add_bool_false_to_true_case() {
    // Test false OR true = true
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([false], &device);
    let indices = TestTensorInt::from_data([0], &device);
    let values = TestTensorBool::<1>::from_data([true], &device);

    let output = tensor.select_assign(0, indices, values, IndexingUpdateOp::Add);
    let expected = TensorData::from([true]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_select_add_bool_true_or_true_accumulation() {
    // Test multiple true accumulations
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([true, false], &device);
    let indices = TestTensorInt::from_data([0, 0, 0], &device);
    let values = TestTensorBool::<1>::from_data([true, true, true], &device);

    let output = tensor.select_assign(0, indices, values, IndexingUpdateOp::Add);
    let expected = TensorData::from([true, false]);

    output.into_data().assert_eq(&expected, false);
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
fn should_select_add_bool_overlapping_indices_vs_default() {
    // Test overlapping indices against default implementation
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([false, true], &device);
    let indices = TestTensorInt::from_data([0, 0], &device);
    let values = TestTensorBool::<1>::from_data([true, false], &device);

    let optimized_result =
        tensor
            .clone()
            .select_assign(0, indices.clone(), values.clone(), IndexingUpdateOp::Add);

    let int_tensor = tensor.int();
    let int_values = values.int();
    let assigned = int_tensor.select_assign(0, indices, int_values, IndexingUpdateOp::Add);
    let default_result = assigned.greater_elem(0);

    optimized_result
        .into_data()
        .assert_eq(&default_result.into_data(), false);
}

#[test]
fn should_select_add_bool_true_or_true_accumulation_vs_default() {
    // Test multiple true accumulations against default implementation
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([true, false], &device);
    let indices = TestTensorInt::from_data([0, 0, 0], &device);
    let values = TestTensorBool::<1>::from_data([true, true, true], &device);

    let optimized_result =
        tensor
            .clone()
            .select_assign(0, indices.clone(), values.clone(), IndexingUpdateOp::Add);

    let int_tensor = tensor.int();
    let int_values = values.int();
    let assigned = int_tensor.select_assign(0, indices, int_values, IndexingUpdateOp::Add);
    let default_result = assigned.greater_elem(0);

    optimized_result
        .into_data()
        .assert_eq(&default_result.into_data(), false);
}

#[test]
fn should_select_add_bool_false_to_true_case_vs_default() {
    // Test false OR true case against default implementation
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([false], &device);
    let indices = TestTensorInt::from_data([0], &device);
    let values = TestTensorBool::<1>::from_data([true], &device);

    let optimized_result =
        tensor
            .clone()
            .select_assign(0, indices.clone(), values.clone(), IndexingUpdateOp::Add);

    let int_tensor = tensor.int();
    let int_values = values.int();
    let assigned = int_tensor.select_assign(0, indices, int_values, IndexingUpdateOp::Add);
    let default_result = assigned.greater_elem(0);

    optimized_result
        .into_data()
        .assert_eq(&default_result.into_data(), false);
}

#[test]
fn should_select_add_bool_tensor_vs_default() {
    // Test existing basic case against default implementation
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([true, false, true], &device);
    let indices = TestTensorInt::from_data([0, 2], &device);
    let values = TestTensorBool::<1>::from_data([false, false], &device);

    let optimized_result =
        tensor
            .clone()
            .select_assign(0, indices.clone(), values.clone(), IndexingUpdateOp::Add);

    let int_tensor = tensor.int();
    let int_values = values.int();
    let assigned = int_tensor.select_assign(0, indices, int_values, IndexingUpdateOp::Add);
    let default_result = assigned.greater_elem(0);

    optimized_result
        .into_data()
        .assert_eq(&default_result.into_data(), false);
}

#[test]
#[should_panic(expected = "Tensors are not eq")]
fn should_fail_if_replacement_semantics_were_used() {
    // Test that framework uses accumulation, not replacement
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([true], &device);
    let indices = TestTensorInt::from_data([0], &device);
    let values = TestTensorBool::<1>::from_data([false], &device);

    let output = tensor.select_assign(0, indices, values, IndexingUpdateOp::Add);
    let replacement_expected = TensorData::from([false]);

    output.into_data().assert_eq(&replacement_expected, false);
}

#[test]
#[should_panic(expected = "Tensors are not eq")]
fn should_fail_if_replacement_semantics_were_used_vs_default() {
    // Test that default implementation also uses accumulation, not replacement
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([true], &device);
    let indices = TestTensorInt::from_data([0], &device);
    let values = TestTensorBool::<1>::from_data([false], &device);

    let int_tensor = tensor.int();
    let int_values = values.int();
    let assigned = int_tensor.select_assign(0, indices, int_values, IndexingUpdateOp::Add);
    let default_result = assigned.greater_elem(0);
    let replacement_expected = TensorData::from([false]);

    default_result
        .into_data()
        .assert_eq(&replacement_expected, false);
}
