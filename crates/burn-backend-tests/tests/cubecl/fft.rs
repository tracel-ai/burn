use super::*;
use burn_tensor::{TensorData, Tolerance};
use burn_tensor::signal::rfft;

#[test]
fn rfft_dim1_sine_wave_produces_imaginary_spectrum() {
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

#[test]
fn rfft_dim1_cosine_wave_produces_real_spectrum() {
    let signal = TestTensor::<2>::from([[
        1.0, 0.7071, 0.0, -0.7071,
        -1.0, -0.7071, 0.0, 0.7071,
    ]]);

    let (spectrum_re, spectrum_im) = rfft(signal, 1);

    let expected_re = TensorData::from([[0.0, 4.0, 0.0, 0.0, 0.0]]);
    let expected_im = TensorData::from([[0.0, 0.0, 0.0, 0.0, 0.0]]);

    spectrum_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));

    spectrum_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn rfft_dim1_2d_tensor_distinct_rows() {
    let signal = TestTensor::<2>::from([
        // freq = 1
        [0.0, 0.7071, 1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071],
        // freq = 2
        [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
    ]);

    let (re, im) = rfft(signal, 1);

    let expected_re = TensorData::from([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]);

    let expected_im = TensorData::from([
        [0.0, -4.0, 0.0, 0.0, 0.0], // freq 1
        [0.0, 0.0, -4.0, 0.0, 0.0], // freq 2
    ]);

    re.into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));

    im.into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}
