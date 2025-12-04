use crate::*;
use burn_tensor::TensorData;
use burn_tensor::backend::Backend;

#[test]
fn test_cartesian_grid() {
    let device = <TestBackend as Backend>::Device::default();

    // Test a single element tensor
    let tensor: TestTensorInt<2> = TestTensorInt::<1>::cartesian_grid([1], &device);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([[0]]), false);

    // Test for a 2x2 tensor
    let tensor: TestTensorInt<3> = TestTensorInt::<2>::cartesian_grid([2, 2], &device);
    tensor.into_data().assert_eq(
        &TensorData::from([[[0, 0], [0, 1]], [[1, 0], [1, 1]]]),
        false,
    );
}
