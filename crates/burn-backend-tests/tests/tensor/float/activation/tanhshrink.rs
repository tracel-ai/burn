use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_tanhshrink_d2() {
    let tensor = TestTensor::<2>::from([[-1.0, 0.0], [1.0, 2.0]]);

    let output = activation::tanhshrink(tensor);
    // tanhshrink(x) = x - tanh(x)
    // -1 - tanh(-1) = -0.238406, 0 - tanh(0) = 0
    // 1 - tanh(1) = 0.238406, 2 - tanh(2) = 1.035972
    let expected = TensorData::from([[-0.238406, 0.0], [0.238406, 1.035972]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
