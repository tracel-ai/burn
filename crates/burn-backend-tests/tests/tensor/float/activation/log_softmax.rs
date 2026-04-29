use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_log_softmax_d2() {
    let tensor = TestTensor::<2>::from([[1.0, 0.0], [0.0, 1.0]]);

    let output = activation::log_softmax(tensor, 1);
    let expected = TensorData::from([[-0.3132617, -1.3132617], [-1.3132617, -0.3132617]]);

    let tolerance = Tolerance::rel_abs(0.01, 0.0001);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}
