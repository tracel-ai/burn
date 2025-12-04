use crate::*;
use burn_tensor::TensorData;
use burn_tensor::backend::Backend;

#[test]
fn test_arange_step() {
    let device = <TestBackend as Backend>::Device::default();

    // Test correct sequence of numbers when the range is 0..9 and the step is 1
    let tensor = TestTensorInt::<1>::arange_step(0..9, 1, &device);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([0, 1, 2, 3, 4, 5, 6, 7, 8]), false);

    // Test correct sequence of numbers when the range is 0..3 and the step is 2
    let tensor = TestTensorInt::<1>::arange_step(0..3, 2, &device);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([0, 2]), false);

    // Test correct sequence of numbers when the range is 0..2 and the step is 5
    let tensor = TestTensorInt::<1>::arange_step(0..2, 5, &device);
    tensor.into_data().assert_eq(&TensorData::from([0]), false);

    // Test correct sequence of numbers when the range includes negative numbers
    let tensor = TestTensorInt::<1>::arange_step(-3..3, 2, &device);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([-3, -1, 1]), false);

    let tensor = TestTensorInt::<1>::arange_step(-5..1, 5, &device);
    tensor
        .clone()
        .into_data()
        .assert_eq(&TensorData::from([-5, 0]), false);
    assert_eq!(tensor.device(), device);
}

#[test]
#[should_panic]
fn should_panic_when_step_is_zero() {
    let device = <TestBackend as Backend>::Device::default();
    // Test that arange_step panics when the step is 0
    let _tensor = TestTensorInt::<1>::arange_step(0..3, 0, &device);
}
