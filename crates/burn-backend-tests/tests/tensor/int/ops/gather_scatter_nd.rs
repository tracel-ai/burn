use super::*;
use burn_tensor::{IndexingUpdateOp, TensorData};

#[test]
fn test_int_gather_nd_2d() {
    let device = Default::default();
    let data = TestTensorInt::<2>::from_ints([[10, 20, 30], [40, 50, 60]], &device);
    // indices shape: [2, 2] => K=2, M=2, DV=1
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [1, 2]], &device);

    let output: TestTensorInt<1> = data.gather_nd(indices);

    output
        .into_data()
        .assert_eq(&TensorData::from([20, 60]), false);
}

#[test]
fn test_int_scatter_nd_add() {
    let device = Default::default();
    let data = TestTensorInt::<2>::from_ints([[1, 2, 3], [4, 5, 6]], &device);
    let indices = TestTensorInt::<2>::from_ints([[0, 0], [1, 2]], &device);
    let values = TestTensorInt::<1>::from_ints([10, 20], &device);

    let output = data.scatter_nd(indices, values, IndexingUpdateOp::Add);

    output
        .into_data()
        .assert_eq(&TensorData::from([[11, 2, 3], [4, 5, 26]]), false);
}

#[test]
fn test_int_scatter_nd_mul() {
    let device = Default::default();
    let data = TestTensorInt::<2>::from_ints([[2, 3], [4, 5]], &device);
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [1, 0]], &device);
    let values = TestTensorInt::<1>::from_ints([10, 3], &device);

    let output = data.scatter_nd(indices, values, IndexingUpdateOp::Mul);

    output
        .into_data()
        .assert_eq(&TensorData::from([[2, 30], [12, 5]]), false);
}

#[test]
fn test_int_scatter_nd_assign() {
    let device = Default::default();
    let data = TestTensorInt::<2>::from_ints([[1, 2, 3], [4, 5, 6]], &device);
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [1, 2]], &device);
    let values = TestTensorInt::<1>::from_ints([99, 88], &device);

    let output = data.scatter_nd(indices, values, IndexingUpdateOp::Assign);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 99, 3], [4, 5, 88]]), false);
}

#[test]
fn test_int_scatter_nd_min() {
    let device = Default::default();
    let data = TestTensorInt::<2>::from_ints([[5, 10], [15, 20]], &device);
    let indices = TestTensorInt::<2>::from_ints([[0, 0], [1, 1]], &device);
    let values = TestTensorInt::<1>::from_ints([3, 25], &device);

    let output = data.scatter_nd(indices, values, IndexingUpdateOp::Min);

    // min(5, 3) = 3; min(20, 25) = 20
    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 10], [15, 20]]), false);
}

#[test]
fn test_int_scatter_nd_max() {
    let device = Default::default();
    let data = TestTensorInt::<2>::from_ints([[5, 10], [15, 20]], &device);
    let indices = TestTensorInt::<2>::from_ints([[0, 0], [1, 1]], &device);
    let values = TestTensorInt::<1>::from_ints([3, 25], &device);

    let output = data.scatter_nd(indices, values, IndexingUpdateOp::Max);

    // max(5, 3) = 5; max(20, 25) = 25
    output
        .into_data()
        .assert_eq(&TensorData::from([[5, 10], [15, 25]]), false);
}

#[test]
fn test_int_scatter_nd_slices() {
    // K=1: each index selects a full row
    let device = Default::default();
    let data = TestTensorInt::<2>::zeros([2, 3], &device);
    let indices = TestTensorInt::<2>::from_ints([[0], [1]], &device);
    let values = TestTensorInt::<2>::from_ints([[10, 20, 30], [40, 50, 60]], &device);

    let output = data.scatter_nd(indices, values, IndexingUpdateOp::Assign);

    output
        .into_data()
        .assert_eq(&TensorData::from([[10, 20, 30], [40, 50, 60]]), false);
}

#[test]
fn test_int_gather_nd_slices() {
    // K=1: each index gathers a full row
    let device = Default::default();
    let data = TestTensorInt::<2>::from_ints([[10, 20, 30], [40, 50, 60]], &device);
    let indices = TestTensorInt::<2>::from_ints([[1], [0]], &device);

    let output: TestTensorInt<2> = data.gather_nd(indices);

    output
        .into_data()
        .assert_eq(&TensorData::from([[40, 50, 60], [10, 20, 30]]), false);
}

#[test]
fn test_int_scatter_nd_single_element() {
    let device = Default::default();
    let data = TestTensorInt::<1>::from_ints([1, 2, 3], &device);
    let indices = TestTensorInt::<2>::from_ints([[0]], &device);
    let values = TestTensorInt::<1>::from_ints([99], &device);

    let output = data.scatter_nd(indices, values, IndexingUpdateOp::Assign);

    output
        .into_data()
        .assert_eq(&TensorData::from([99, 2, 3]), false);
}
