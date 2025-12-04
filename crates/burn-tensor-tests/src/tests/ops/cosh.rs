use crate::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_cosh_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.cosh();
    let expected = TensorData::from([[1.0000, 1.5431, 3.7622], [10.0677, 27.3082, 74.2099]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
