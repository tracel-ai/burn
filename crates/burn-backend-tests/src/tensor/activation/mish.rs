use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_mish() {
    let tensor = TestTensor::<2>::from([[-0.4240, -0.9574, -0.2215], [-0.5767, 0.7218, -0.1620]]);

    let output = activation::mish(tensor);
    let expected = TensorData::from([
        [-0.19709, -0.30056, -0.11714],
        [-0.24132, 0.58235, -0.08877],
    ]);

    // Metal has less precise trigonometric functions (tanh inside mish)
    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}
