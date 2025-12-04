use crate::*;
use burn_tensor::Tolerance;
use burn_tensor::{ TensorData};

#[test]
fn should_support_exp_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.exp();
    let expected = TensorData::from([[1.0, 2.71830, 7.3891], [20.0855, 54.5981, 148.4132]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
