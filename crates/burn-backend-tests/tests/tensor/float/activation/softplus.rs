use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_softplus_large_values_do_not_overflow() {
    // `exp` overflows above ~88.7 in f32, so a naive `log(exp(beta * x) + 1)` returns `inf`
    // here. For large `beta * x`, softplus converges to the identity.
    let tensor = TestTensor::<1>::from([20.0, 50.0, 100.0, 1000.0]);

    let output = activation::softplus(tensor.clone(), 1.0, 20.0);
    let expected = TensorData::from([20.0, 50.0, 100.0, 1000.0]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    // `beta` scales the input, so `beta = 5` overflows from `x = ~18` upwards.
    let output = activation::softplus(tensor, 5.0, 20.0);
    let expected = TensorData::from([20.0, 50.0, 100.0, 1000.0]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_softplus_small_values_do_not_underflow() {
    // For very negative `beta * x`, softplus converges to `exp(beta * x) / beta`.
    let tensor = TestTensor::<1>::from([-20.0, -50.0]);

    let output = activation::softplus(tensor, 1.0, 20.0);
    let expected = TensorData::from([2.0611537e-9, 1.9287499e-22]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_softplus_with_threshold() {
    let tensor = TestTensor::<1>::from([5.0, 25.0]);

    // The default threshold of 20 evaluates the first element and substitutes the second.
    let output = activation::softplus(tensor.clone(), 1.0, 20.0);
    let expected = TensorData::from([5.0067153, 25.0]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    // Raising it past both evaluates both, which is still accurate well beyond the default.
    let output = activation::softplus(tensor.clone(), 1.0, 80.0);
    let expected = TensorData::from([5.0067153, 25.0]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    // Lowering it below both substitutes the identity for both.
    let output = activation::softplus(tensor, 1.0, 1.0);
    let expected = TensorData::from([5.0, 25.0]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_softplus_d2() {
    let tensor = TestTensor::<2>::from([[-0.4240, -0.9574, -0.2215], [-0.5767, 0.7218, -0.1620]]);

    let output = activation::softplus(tensor.clone(), 1.0, 20.0);
    let expected = TensorData::from([
        [0.503453, 0.324898, 0.588517],
        [0.445806, 1.117805, 0.615424],
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let output = activation::softplus(tensor, 2.0, 20.0);
    let expected = TensorData::from([
        [0.178232, 0.068737, 0.247990],
        [0.137132, 0.827771, 0.272106],
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
