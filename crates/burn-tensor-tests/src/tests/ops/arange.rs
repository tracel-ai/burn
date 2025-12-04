use crate::*;
use burn_tensor::TensorData;
use burn_tensor::backend::Backend;

#[test]
fn test_arange() {
    let device = <TestBackend as Backend>::Device::default();

    let tensor = TestTensorInt::<1>::arange(2..5, &device);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([2, 3, 4]), false);

    // Test arange with negative numbers
    let tensor = TestTensorInt::<1>::arange(-10..-5, &device);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([-10, -9, -8, -7, -6]), false);

    let tensor = TestTensorInt::<1>::arange(-3..0, &device);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([-3, -2, -1]), false);

    // Test arange with a mix of positive and negative numbers
    let tensor = TestTensorInt::<1>::arange(-2..3, &device);
    tensor
        .clone()
        .into_data()
        .assert_eq(&TensorData::from([-2, -1, 0, 1, 2]), false);
    assert_eq!(tensor.device(), device);
}
