use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_relu6_d2() {
    let tensor = TestTensor::<2>::from([[-2.0, 0.0, 3.0], [6.0, 8.0, 5.5]]);

    let output = activation::relu6(tensor);
    // relu6(x) = min(max(0, x), 6)
    // relu6(-2) = 0, relu6(0) = 0, relu6(3) = 3
    // relu6(6) = 6, relu6(8) = 6 (clamped), relu6(5.5) = 5.5
    let expected = TensorData::from([[0.0, 0.0, 3.0], [6.0, 6.0, 5.5]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_relu6_at_boundaries() {
    let tensor = TestTensor::<1>::from([-0.1, 0.0, 6.0, 6.1]);

    let output = activation::relu6(tensor);
    // exactly-zero and exactly-six inputs map to themselves; just outside clamps.
    let expected = TensorData::from([0.0, 0.0, 6.0, 6.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
