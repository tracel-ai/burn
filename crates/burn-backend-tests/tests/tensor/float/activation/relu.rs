use super::*;
use burn_tensor::{TensorData, Tolerance, activation};

#[test]
fn test_relu_d2() {
    let tensor = TestTensor::<2>::from([[0.0, -1.0, 2.0], [3.0, -4.0, 5.0]]);

    let output = activation::relu(tensor);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0.0, 0.0, 2.0], [3.0, 0.0, 5.0]]), false);
}

#[test]
fn test_relu_d1() {
    let tensor = TestTensor::<1>::from([-2.0, -1.0, 0.0, 1.0, 2.0]);

    let output = activation::relu(tensor);

    output.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([0.0, 0.0, 0.0, 1.0, 2.0]),
        Tolerance::absolute(1e-6),
    );
}
