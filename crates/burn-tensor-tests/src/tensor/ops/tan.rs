use crate::*;
use burn_tensor::Tolerance;
use burn_tensor::{ TensorData};

#[test]
fn should_support_tan_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.tan();
    let expected = TensorData::from([[0.0, 1.557408, -2.185040], [-0.142547, 1.157821, -3.380515]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
