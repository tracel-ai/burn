use crate::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_tanh() {
    let tensor = TestTensor::<2>::from([[1., 2.], [3., 4.]]);

    let output = activation::tanh(tensor);
    let expected = TensorData::from([[0.761594, 0.964028], [0.995055, 0.999329]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
