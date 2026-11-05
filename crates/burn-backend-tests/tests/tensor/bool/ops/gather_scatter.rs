use super::*;
use burn_tensor::{IndexingUpdateOp, TensorData};

#[test]
fn should_scatter_1d_bool() {
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([true, false, false], &device);
    let values = TestTensorBool::from_data([false, true, true], &device);
    let indices = TestTensorInt::from_ints([1, 0, 2], &device);

    let output = tensor.scatter(0, indices, values, IndexingUpdateOp::Add);

    output
        .into_data()
        .assert_eq(&TensorData::from([true, false, true]), false);
}

#[test]
fn should_gather_1d_dim0_bool() {
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([true, false, false], &device);
    let indices = TestTensorInt::from_ints([1, 1, 0, 1, 2], &device);

    let output = tensor.gather(0, indices);

    output
        .into_data()
        .assert_eq(&TensorData::from([false, false, true, false, false]), false);
}

/// Scatters into `[true, true, false, false]` so each position pairs a distinct
/// (current, update) combination: (T, T), (T, F), (F, T), (F, F).
fn scatter_bool_truth_table(update: IndexingUpdateOp) -> TensorData {
    let device = Default::default();
    let tensor = TestTensorBool::<1>::from_data([true, true, false, false], &device);
    let indices = TestTensorInt::from_ints([1, 0, 3, 2], &device);
    let values = TestTensorBool::from_data([false, true, false, true], &device);

    tensor.scatter(0, indices, values, update).into_data()
}

#[test]
fn should_scatter_assign_bool() {
    scatter_bool_truth_table(IndexingUpdateOp::Assign)
        .assert_eq(&TensorData::from([true, false, true, false]), false);
}

#[test]
fn should_scatter_or_bool_for_add_and_max() {
    let expected = TensorData::from([true, true, true, false]);
    scatter_bool_truth_table(IndexingUpdateOp::Add).assert_eq(&expected, false);
    scatter_bool_truth_table(IndexingUpdateOp::Max).assert_eq(&expected, false);
}

#[test]
fn should_scatter_and_bool_for_mul_and_min() {
    let expected = TensorData::from([true, false, false, false]);
    scatter_bool_truth_table(IndexingUpdateOp::Mul).assert_eq(&expected, false);
    scatter_bool_truth_table(IndexingUpdateOp::Min).assert_eq(&expected, false);
}

#[test]
fn should_scatter_assign_bool_2d_dim1() {
    let device = Default::default();
    let tensor =
        TestTensorBool::<2>::from_data([[true, true, true], [false, false, false]], &device);
    let indices = TestTensorInt::from_ints([[2, 0], [1, 2]], &device);
    let values = TestTensorBool::from_data([[false, false], [true, true]], &device);

    let output = tensor.scatter(1, indices, values, IndexingUpdateOp::Assign);

    output.into_data().assert_eq(
        &TensorData::from([[false, true, false], [false, true, true]]),
        false,
    );
}
