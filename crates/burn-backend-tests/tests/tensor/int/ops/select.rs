use super::*;
use burn_tensor::{IndexingUpdateOp, TensorData};

#[test]
fn should_select_1d_int() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::from_data([5, 6, 7], &device);
    let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &device);

    let output = tensor.select(0, indices);
    let expected = TensorData::from([6, 6, 5, 6, 7]);

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

#[cfg(feature = "ndarray")]
#[test]
fn should_select_assign_2d_dim0_int() {
    let device = Default::default();
    let tensor = TestTensorInt::<2>::from_data([[10, 20], [30, 40], [50, 60]], &device);
    let values = TestTensorInt::from_data([[5, 70], [80, 1]], &device);
    let indices = TestTensorInt::from_data(TensorData::from([2, 0]), &device);

    let output = tensor.select_assign(0, indices, values, IndexingUpdateOp::Assign);
    let expected = TensorData::from([[80, 1], [30, 40], [5, 70]]);

    output.into_data().assert_eq(&expected, false);
}

#[cfg(feature = "ndarray")]
#[test]
fn should_select_assign_2d_dim1_int() {
    let device = Default::default();
    let tensor = TestTensorInt::<2>::from_data([[10, 20, 30], [40, 50, 60]], &device);
    let values = TestTensorInt::from_data([[7, 8], [9, 10]], &device);
    let indices = TestTensorInt::from_data(TensorData::from([2, 0]), &device);

    let output = tensor.select_assign(1, indices, values, IndexingUpdateOp::Assign);
    let expected = TensorData::from([[8, 20, 7], [10, 50, 9]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic]
fn should_panic_select_add_invalid_num_indices() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::from_data([0; 12], &device);
    let values = TestTensorInt::from_data([1; 12], &device);
    let indices = TestTensorInt::from_data(TensorData::from([1]), &device);

    tensor.select_assign(0, indices, values, IndexingUpdateOp::Add);
}
