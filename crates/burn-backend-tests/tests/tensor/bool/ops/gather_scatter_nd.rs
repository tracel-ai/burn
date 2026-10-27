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

/// Scatters `count` updates onto each of two positions: position 0 starts false and gets
/// one true among falses, position 1 starts true and gets one false among trues. With
/// repeated targets, the or (`Add`/`Max`) and the and (`Mul`/`Min`) must still see the
/// single differing update, whatever order the updates land in.
fn scatter_nd_bool_repeated_targets(update: IndexingUpdateOp) -> TensorData {
    let device = Default::default();
    let count = 2048;
    let data = TestTensorBool::<1>::from_data([false, true], &device);
    let mut indices = Vec::with_capacity(2 * count);
    let mut values = Vec::with_capacity(2 * count);
    for i in 0..count {
        indices.push([0i64]);
        values.push(i == count / 2);
        indices.push([1i64]);
        values.push(i != count / 2);
    }
    let indices =
        TestTensorInt::<2>::from_data(TensorData::new(indices.concat(), [2 * count, 1]), &device);
    let values = TestTensorBool::<1>::from_data(TensorData::new(values, [2 * count]), &device);

    data.scatter_nd(indices, values, update).into_data()
}

#[test]
fn test_bool_scatter_nd_or_with_repeated_targets() {
    // Position 0 gets one true, so the or is true; position 1 starts true
    let expected = TensorData::from([true, true]);
    for update in [IndexingUpdateOp::Add, IndexingUpdateOp::Max] {
        scatter_nd_bool_repeated_targets(update).assert_eq(&expected, false);
    }
}

#[test]
fn test_bool_scatter_nd_and_with_repeated_targets() {
    // Position 0 starts false; position 1 gets one false, so the and is false
    let expected = TensorData::from([false, false]);
    for update in [IndexingUpdateOp::Mul, IndexingUpdateOp::Min] {
        scatter_nd_bool_repeated_targets(update).assert_eq(&expected, false);
    }
}

/// Scatters whole rows: row 0 gets two updates, row 1 none, and row 2 one that never holds
/// the absorbing value, so most positions must keep their data value.
fn scatter_nd_bool_rows(
    data: [[bool; 2]; 3],
    values: [[bool; 2]; 3],
    update: IndexingUpdateOp,
) -> TensorData {
    let device = Default::default();
    let data = TestTensorBool::<2>::from_data(data, &device);
    let indices = TestTensorInt::<2>::from_ints([[0], [0], [2]], &device);
    let values = TestTensorBool::<2>::from_data(values, &device);

    data.scatter_nd(indices, values, update).into_data()
}

#[test]
fn test_bool_scatter_nd_or_keeps_unset_values() {
    let data = [[false, false], [false, true], [true, false]];
    let values = [[false, false], [false, true], [false, false]];
    let expected = TensorData::from([[false, true], [false, true], [true, false]]);
    for update in [IndexingUpdateOp::Add, IndexingUpdateOp::Max] {
        scatter_nd_bool_rows(data, values, update).assert_eq(&expected, false);
    }
}

#[test]
fn test_bool_scatter_nd_and_keeps_unset_values() {
    let data = [[true, true], [false, true], [true, false]];
    let values = [[true, true], [true, false], [true, true]];
    let expected = TensorData::from([[true, false], [false, true], [true, false]]);
    for update in [IndexingUpdateOp::Mul, IndexingUpdateOp::Min] {
        scatter_nd_bool_rows(data, values, update).assert_eq(&expected, false);
    }
}

/// Scatters rows of a transposed (strided) view of `base`.
fn scatter_nd_bool_strided(
    base: [[bool; 3]; 2],
    indices: [[i64; 1]; 2],
    values: [[bool; 2]; 2],
    update: IndexingUpdateOp,
) -> TensorData {
    let device = Default::default();
    let data = TestTensorBool::<2>::from_data(base, &device).transpose();
    let indices = TestTensorInt::<2>::from_ints(indices, &device);
    let values = TestTensorBool::<2>::from_data(values, &device);

    data.scatter_nd(indices, values, update).into_data()
}

#[test]
fn test_bool_scatter_nd_or_strided_data() {
    // data = [[false, true], [true, false], [false, false]]
    let base = [[false, true, false], [true, false, false]];
    let values = [[false, true], [true, false]];
    let expected = TensorData::from([[false, true], [true, true], [true, false]]);
    for update in [IndexingUpdateOp::Add, IndexingUpdateOp::Max] {
        scatter_nd_bool_strided(base, [[1], [2]], values, update).assert_eq(&expected, false);
    }
}

#[test]
fn test_bool_scatter_nd_and_strided_data() {
    // data = [[true, true], [true, true], [false, true]]
    let base = [[true, true, false], [true, true, true]];
    let values = [[true, false], [false, true]];
    let expected = TensorData::from([[true, false], [true, true], [false, true]]);
    for update in [IndexingUpdateOp::Mul, IndexingUpdateOp::Min] {
        scatter_nd_bool_strided(base, [[0], [2]], values, update).assert_eq(&expected, false);
    }
}
