use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_ceil_ops() {
    let data = TensorData::from([[24.0423, 87.9478, 76.1838], [59.6929, 43.8169, 94.8826]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.ceil();
    let expected = TensorData::from([[25., 88., 77.], [60., 44., 95.]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
