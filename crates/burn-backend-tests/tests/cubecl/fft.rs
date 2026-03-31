use super::*;
use burn_tensor::module::rfft;
use burn_tensor::{DType, TensorData, Tolerance};

#[test]
fn rfft_of_tensor_dim2() {
    let signal = TestTensor::<2>::from([[0.0, 1.2071, 1.0, 0.2071, 0.0, -0.2071, -1.0, -1.2071]]);
    let dim = 1;
    let (spectrum_re, spectrum_im) = rfft(signal.clone(), dim);
    let expected_re = TensorData::from([[0, 0, 0, 0, 0]]);
    let expected_im = TensorData::from([[0, -4, -2, 0, 0]]);
    spectrum_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-4));

    spectrum_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-4));
}
