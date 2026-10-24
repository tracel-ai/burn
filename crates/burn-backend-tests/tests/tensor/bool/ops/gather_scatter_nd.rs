use super::*;
use burn_tensor::{IndexingUpdateOp, TensorData};

#[test]
fn test_bool_gather_nd_elements() {
    let device = Default::default();
    let data = TestTensorBool::<2>::from_data([[true, false], [false, true]], &device);
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [1, 1], [0, 0]], &device);

    let output: TestTensorBool<1> = data.gather_nd(indices);

    output
        .into_data()
        .assert_eq(&TensorData::from([false, true, true]), false);
}

#[test]
fn test_bool_gather_nd_slices() {
    let device = Default::default();
    let data = TestTensorBool::<2>::from_data([[true, false], [false, true]], &device);
    let indices = TestTensorInt::<2>::from_ints([[1]], &device);

    let output: TestTensorBool<2> = data.gather_nd(indices);

    output
        .into_data()
        .assert_eq(&TensorData::from([[false, true]]), false);
}

/// Scatters into `[[true, true], [false, false]]` so each position pairs a distinct
/// (current, update) combination: (T, T), (T, F), (F, T), (F, F).
fn scatter_nd_bool_truth_table(update: IndexingUpdateOp) -> TensorData {
    let device = Default::default();
    let data = TestTensorBool::<2>::from_data([[true, true], [false, false]], &device);
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [0, 0], [1, 1], [1, 0]], &device);
    let values = TestTensorBool::<1>::from_data([false, true, false, true], &device);

    data.scatter_nd(indices, values, update).into_data()
}

#[test]
fn test_bool_scatter_nd_assign() {
    scatter_nd_bool_truth_table(IndexingUpdateOp::Assign)
        .assert_eq(&TensorData::from([[true, false], [true, false]]), false);
}

#[test]
fn test_bool_scatter_nd_or_for_add_and_max() {
    let expected = TensorData::from([[true, true], [true, false]]);
    scatter_nd_bool_truth_table(IndexingUpdateOp::Add).assert_eq(&expected, false);
    scatter_nd_bool_truth_table(IndexingUpdateOp::Max).assert_eq(&expected, false);
}

#[test]
fn test_bool_scatter_nd_and_for_mul_and_min() {
    let expected = TensorData::from([[true, false], [false, false]]);
    scatter_nd_bool_truth_table(IndexingUpdateOp::Mul).assert_eq(&expected, false);
    scatter_nd_bool_truth_table(IndexingUpdateOp::Min).assert_eq(&expected, false);
}

#[test]
fn test_bool_scatter_nd_assign_slices() {
    let device = Default::default();
    let data = TestTensorBool::<2>::from_data([[true, true], [false, false]], &device);
    let indices = TestTensorInt::<2>::from_ints([[1]], &device);
    let values = TestTensorBool::<2>::from_data([[true, false]], &device);

    let output = data.scatter_nd(indices, values, IndexingUpdateOp::Assign);

    output
        .into_data()
        .assert_eq(&TensorData::from([[true, true], [true, false]]), false);
}
