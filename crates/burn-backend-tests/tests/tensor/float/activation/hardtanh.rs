use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_hardtanh_default_d2() {
    let tensor = TestTensor::<2>::from([[-2.0, -1.0, 0.0], [0.5, 1.0, 2.0]]);

    let output = activation::hardtanh(tensor, -1.0, 1.0);
    // hardtanh(x, -1, 1) = clamp(x, -1, 1)
    let expected = TensorData::from([[-1.0, -1.0, 0.0], [0.5, 1.0, 1.0]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_hardtanh_custom_range() {
    let tensor = TestTensor::<1>::from([-3.0, -2.0, 0.0, 2.0, 3.0]);

    let output = activation::hardtanh(tensor, -2.0, 2.0);
    // hardtanh(x, -2, 2) = clamp(x, -2, 2)
    let expected = TensorData::from([-2.0, -2.0, 0.0, 2.0, 2.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
