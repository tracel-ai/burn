use crate::*;
use burn_tensor::TensorData;

#[test]
fn test_float() {
    let device = Default::default();
    let tensor_1 = TestTensor::<1>::from_data([1.0, 2.0, 3.0], &device);
    let tensor_2 = TestTensor::<1>::from_data([0.0, -1.0, 4.0], &device);

    let output = tensor_1.dot(tensor_2);
    let expected = TensorData::from([10.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int() {
    let device = Default::default();
    let tensor_1 = TestTensor::<1>::from_data([1, 2, 3], &device);
    let tensor_2 = TestTensor::<1>::from_data([0, -1, 4], &device);

    let output = tensor_1.dot(tensor_2);
    let expected = TensorData::from([10]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic]
fn test_panics_for_different_sizes() {
    let device = Default::default();
    let tensor_1 = TestTensor::<1>::from_data([1, 2], &device);
    let tensor_2 = TestTensor::<1>::from_data([1, 2, 3], &device);
    let _output = tensor_1.dot(tensor_2);
}
