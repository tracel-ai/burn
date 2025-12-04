use crate::*;
use burn_tensor::Tolerance;
use burn_tensor::{ TensorData};

#[test]
fn should_support_exp_log1p() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.log1p();
    let expected = TensorData::from([
        [0.0, core::f32::consts::LN_2, 1.09861],
        [1.38629, 1.60944, 1.79176],
    ]);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::default().set_half_precision_relative(1e-3),
    );
}
