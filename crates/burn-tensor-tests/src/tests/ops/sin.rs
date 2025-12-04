use crate::*;
use burn_tensor::Tolerance;
use burn_tensor::{ TensorData};

#[test]
fn should_support_sin_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.sin();
    let expected = TensorData::from([[0.0, 0.841471, 0.909297], [0.141120, -0.756802, -0.958924]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
