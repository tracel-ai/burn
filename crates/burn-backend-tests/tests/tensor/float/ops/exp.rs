use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use core::f32::consts::E;

#[test]
fn should_support_exp_transposed() {
    // [[0, 1], [2, 3]] transposed -> [[0, 2], [1, 3]]
    let tensor = TestTensor::<2>::from([[0.0, 1.0], [2.0, 3.0]]);
    let transposed = tensor.transpose();

    let output = transposed.exp();
    let expected = TensorData::from([[1.0, E * E], [E, E * E * E]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_exp_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.exp();
    let expected = TensorData::from([[1.0, 2.71830, 7.3891], [20.0855, 54.5981, 148.4132]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
