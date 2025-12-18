use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_round_ops() {
    let data = TensorData::from([[24.0423, 87.9478, 76.1838], [59.6929, 43.8169, 94.8826]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.round();
    let expected = TensorData::from([[24., 88., 76.], [60., 44., 95.]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_round_ties_even() {
    let data = TensorData::from([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());

    let output = tensor.round();
    let expected = TensorData::from([2., 2., 4., 4., 6., 6.]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
