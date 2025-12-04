use crate::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_silu() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

    let output = activation::silu(tensor);
    let expected = TensorData::from([[0.73106, 1.76159], [2.85772, 3.92806]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
