use crate::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_softmin_d2() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

    let output = activation::softmin(tensor, 1);
    let expected = TensorData::from([[9.975274e-01, 2.472623e-03], [1.125352e-07, 1.0000]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
