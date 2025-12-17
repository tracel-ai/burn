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
