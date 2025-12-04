use super::*;
use crate::qtensor::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn flip_float() {
    let tensor = QTensor::<TestBackend, 3>::int8([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]]);

    let flipped = tensor.clone().flip([0, 2]);
    let expected = TensorData::from([[[5., 4., 3.]], [[2., 1., 0.]]]);

    flipped
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));

    // Test with no flip
    let flipped = tensor.clone().flip([]);
    tensor.into_data().assert_eq(&flipped.into_data(), true);
}

#[test]
#[should_panic]
fn flip_duplicated_axes() {
    let tensor = QTensor::<TestBackend, 3>::int8([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]]);

    // Test with a duplicated axis
    let _ = tensor.flip([0, 0, 1]);
}

#[test]
#[should_panic]
fn flip_out_of_bound_axis() {
    let tensor = QTensor::<TestBackend, 3>::int8([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]]);

    // Test with an out of bound axis
    let _ = tensor.clone().flip([3, 0, 1]);
}
