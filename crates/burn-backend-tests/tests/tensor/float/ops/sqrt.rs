use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use core::f32::consts::SQRT_2;

#[test]
fn should_support_sqrt_narrowed() {
    // [1, 4, 9, 16, 25, 36] narrowed to [4, 9, 16, 25]
    let tensor = TestTensor::<1>::from([1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    let narrowed = tensor.narrow(0, 1, 4);

    let output = narrowed.sqrt();
    let expected = TensorData::from([2.0, 3.0, 4.0, 5.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_sqrt_flipped_2d() {
    // [[1, 4], [9, 16]] flipped on axis 0 -> [[9, 16], [1, 4]]
    let tensor = TestTensor::<2>::from([[1.0, 4.0], [9.0, 16.0]]);
    let flipped = tensor.flip([0]);

    let output = flipped.sqrt();
    let expected = TensorData::from([[3.0, 4.0], [1.0, 2.0]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_sqrt_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.sqrt();
    let expected = TensorData::from([[0.0, 1.0, SQRT_2], [1.73205, 2.0, 2.2360]]);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::relative(1e-4).set_half_precision_relative(1e-3),
    );
}
