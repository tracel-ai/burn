use super::*;
use burn_tensor::TensorData;

// Skip on metal - shader compilation fails for fused int remainder
#[cfg(not(feature = "metal"))]
#[test]
fn should_support_int_remainder_basic() {
    let data = TensorData::from([-3, -2, -1, 1, 2, 3]);
    let device = Default::default();
    let lhs = TestTensorInt::<1>::from_data(data, &device);

    let rhs = TestTensorInt::from_data(TensorData::from([2, 3, 1, 2, 1, 3]), &device);
    let output = lhs.remainder(rhs);
    let expected = TensorData::from([1, 1, -0, 1, 0, 0]);

    output.into_data().assert_eq(&expected, false);
}

// Skip on metal - shader compilation fails for fused int remainder
#[cfg(not(feature = "metal"))]
#[test]
fn should_support_int_remainder_basic_scalar() {
    let data = TensorData::from([-3, -2, -1, 1, 2, 3]);
    let device = Default::default();
    let tensor = TestTensorInt::<1>::from_data(data, &device);

    let output = tensor.remainder_scalar(2);
    let expected = TensorData::from([1, 0, 1, 1, 0, 1]);

    output.into_data().assert_eq(&expected, false);
}
