use super::*;
use burn_tensor::signal::{irfft, rfft};
use burn_tensor::{TensorData, Tolerance};

#[test]
fn rfft_zeros() {
    let signal = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0]);
    let (re, im) = rfft(signal, 0);

    let expected_re = TensorData::from([0.0, 0.0, 0.0]);
    let expected_im = TensorData::from([0.0, 0.0, 0.0]);

    re.into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-4));
    im.into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-4));
}

#[test]
fn rfft_constant() {
    let signal = TestTensor::<1>::from([1.0, 1.0, 1.0, 1.0]);
    let (re, im) = rfft(signal, 0);

    let expected_re = TensorData::from([4.0, 0.0, 0.0]);
    let expected_im = TensorData::from([0.0, 0.0, 0.0]);

    re.into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-4));
    im.into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-4));
}

#[test]
fn rfft_length1() {
    let signal = TestTensor::<1>::from([5.0]);
    let (re, im) = rfft(signal, 0);

    let expected_re = TensorData::from([5.0]);
    let expected_im = TensorData::from([0.0]);

    re.into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-4));
    im.into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-4));
}

#[test]
fn rfft_length2() {
    let signal = TestTensor::<1>::from([1.0, -1.0]);
    let (re, im) = rfft(signal, 0);

    let expected_re = TensorData::from([0.0, 2.0]);
    let expected_im = TensorData::from([0.0, 0.0]);

    re.into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-4));
    im.into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-4));
}

#[test]
fn rfft_dim1_sine_wave_produces_imaginary_spectrum() {
    let signal = TestTensor::<2>::from([[0.0, 1.2071, 1.0, 0.2071, 0.0, -0.2071, -1.0, -1.2071]]);
    let dim = 1;
    let (spectrum_re, spectrum_im) = rfft(signal.clone(), dim);
    let expected_re = TensorData::from([[0, 0, 0, 0, 0]]);
    let expected_im = TensorData::from([[0, -4, -2, 0, 0]]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), expected_re.shape);

    spectrum_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-4));

    spectrum_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-4));
}

#[test]
fn rfft_dim1_cosine_wave_produces_real_spectrum() {
    let signal = TestTensor::<2>::from([[1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071, 0.0, 0.7071]]);

    let (spectrum_re, spectrum_im) = rfft(signal, 1);

    let expected_re = TensorData::from([[0.0, 4.0, 0.0, 0.0, 0.0]]);
    let expected_im = TensorData::from([[0.0, 0.0, 0.0, 0.0, 0.0]]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), expected_re.shape);

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
        [0.0, 0.7071, 1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071],
        [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
    ]);

    let (spectrum_re, spectrum_im) = rfft(signal, 1);

    let expected_re = TensorData::from([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]);

    let expected_im = TensorData::from([[0.0, -4.0, 0.0, 0.0, 0.0], [0.0, 0.0, -4.0, 0.0, 0.0]]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), expected_re.shape);

    spectrum_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));

    spectrum_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn rfft_dim0_2d_tensor() {
    let signal = TestTensor::<2>::from([
        [0.0, 0.0],
        [0.7071, 1.0],
        [1.0, 0.0],
        [0.7071, -1.0],
        [0.0, 0.0],
        [-0.7071, 1.0],
        [-1.0, 0.0],
        [-0.7071, -1.0],
    ]);

    let (spectrum_re, spectrum_im) = rfft(signal, 0);

    let expected_re =
        TensorData::from([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);

    let expected_im =
        TensorData::from([[0.0, 0.0], [-4.0, 0.0], [0.0, -4.0], [0.0, 0.0], [0.0, 0.0]]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), expected_re.shape);

    spectrum_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));

    spectrum_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn rfft_dim2_3d_tensor() {
    let signal = TestTensor::<3>::from([
        [
            [0.0, 0.7071, 1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071],
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
        ],
        [
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
            [0.0, 0.7071, 1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071],
        ],
    ]);

    let (spectrum_re, spectrum_im) = rfft(signal, 2);

    let expected_re = TensorData::from([
        [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
    ]);

    let expected_im = TensorData::from([
        [[0.0, -4.0, 0.0, 0.0, 0.0], [0.0, 0.0, -4.0, 0.0, 0.0]],
        [[0.0, 0.0, -4.0, 0.0, 0.0], [0.0, -4.0, 0.0, 0.0, 0.0]],
    ]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), expected_re.shape);

    spectrum_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));

    spectrum_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn rfft_dim1_3d_tensor() {
    let signal = TestTensor::<3>::from([
        [[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0], [0.0, -1.0], [0.0, 0.0]],
    ]);

    let (spectrum_re, spectrum_im) = rfft(signal, 1);

    let expected_re = TensorData::from([
        [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
    ]);

    let expected_im = TensorData::from([
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), expected_re.shape);

    spectrum_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));

    spectrum_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn irfft_dim1_imaginary_spectrum_produces_sine_wave() {
    let spectrum_re = TestTensor::<2>::from([[0.0, 0.0, 0.0, 0.0, 0.0]]);
    let spectrum_im = TestTensor::<2>::from([[0.0, -4.0, -2.0, 0.0, 0.0]]);

    let signal = irfft(spectrum_re, spectrum_im, 1);

    let expected = TensorData::from([[0.0, 1.2071, 1.0, 0.2071, 0.0, -0.2071, -1.0, -1.2071]]);

    signal
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-3));
}

#[test]
fn irfft_dim1_real_spectrum_produces_cosine_wave() {
    let spectrum_re = TestTensor::<2>::from([[0.0, 4.0, 0.0, 0.0, 0.0]]);
    let spectrum_im = TestTensor::<2>::from([[0.0, 0.0, 0.0, 0.0, 0.0]]);

    let signal = irfft(spectrum_re, spectrum_im, 1);

    let expected = TensorData::from([[1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071, 0.0, 0.7071]]);

    signal
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-3));
}

#[test]
fn irfft_dim1_2d_tensor_distinct_rows() {
    let spectrum_re = TestTensor::<2>::from([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]);

    let spectrum_im =
        TestTensor::<2>::from([[0.0, -4.0, 0.0, 0.0, 0.0], [0.0, 0.0, -4.0, 0.0, 0.0]]);

    let signal = irfft(spectrum_re, spectrum_im, 1);

    let expected = TensorData::from([
        [0.0, 0.7071, 1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071],
        [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
    ]);

    signal
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-3));
}

#[test]
fn irfft_dim0_2d_tensor() {
    let spectrum_re =
        TestTensor::<2>::from([[0.0, 0.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);

    let spectrum_im =
        TestTensor::<2>::from([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);

    let signal = irfft(spectrum_re, spectrum_im, 0);

    let expected = TensorData::from([
        [1.0, 1.0],
        [0.7071, 0.7071],
        [0.0, 0.0],
        [-0.7071, -0.7071],
        [-1.0, -1.0],
        [-0.7071, -0.7071],
        [0.0, 0.0],
        [0.7071, 0.7071],
    ]);

    signal
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-3));
}

#[test]
fn rfft_irfft_roundtrip_1d() {
    let signal = TestTensor::<1>::from([0.0, 1.2071, 1.0, 0.2071, 0.0, -0.2071, -1.0, -1.2071]);

    let (re, im) = rfft(signal.clone(), 0);
    let reconstructed = irfft(re, im, 0);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&signal.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn rfft_irfft_roundtrip_dim1_2d() {
    let signal = TestTensor::<2>::from([
        [0.0, 0.7071, 1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071],
        [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
    ]);

    let (re, im) = rfft(signal.clone(), 1);
    let reconstructed = irfft(re, im, 1);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&signal.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn rfft_irfft_roundtrip_dim0_2d() {
    let signal = TestTensor::<2>::from([
        [1.0, 0.0],
        [0.7071, 1.0],
        [0.0, 0.0],
        [-0.7071, -1.0],
        [-1.0, 0.0],
        [-0.7071, 1.0],
        [0.0, 0.0],
        [0.7071, -1.0],
    ]);

    let (re, im) = rfft(signal.clone(), 0);
    let reconstructed = irfft(re, im, 0);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&signal.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn rfft_irfft_roundtrip_dim1_3d() {
    let signal = TestTensor::<3>::from([
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
        ],
        [
            [1.0, 0.0],
            [0.7071, 1.0],
            [0.0, 0.0],
            [-0.7071, -1.0],
            [-1.0, 0.0],
            [-0.7071, 1.0],
            [0.0, 0.0],
            [0.7071, -1.0],
        ],
    ]);

    let (re, im) = rfft(signal.clone(), 1);
    let reconstructed = irfft(re, im, 1);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&signal.into_data(), Tolerance::absolute(1e-3));
}
