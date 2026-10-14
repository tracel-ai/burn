use super::*;
use burn_core::tensor::{TensorData, Tolerance};
use burn_signal::{cfft, ifft, irfft, rfft};

const SQ2INV: f64 = std::f64::consts::FRAC_1_SQRT_2;
const SQ2INV_PLUS_HALF: f64 = SQ2INV + 0.5;
const SQ2INV_MINUS_HALF: f64 = SQ2INV - 0.5;

#[test]
fn rfft_zeros() {
    let signal = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0]);
    let (re, im) = rfft(signal, 0, None);

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
    let (re, im) = rfft(signal, 0, None);

    let expected_re = TensorData::from([4.0, 0.0, 0.0]);
    let expected_im = TensorData::from([0.0, 0.0, 0.0]);

    re.into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-4));
    im.into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-4));
}

#[test]
fn rfft_length1() {
    assert_length_one_behavior(|| {
        let signal = TestTensor::<1>::from([5.0]);
        let (re, im) = rfft(signal, 0, None);

        let expected_re = TensorData::from([5.0]);
        let expected_im = TensorData::from([0.0]);

        re.into_data()
            .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-4));
        im.into_data()
            .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-4));
    });
}

#[test]
fn rfft_length2() {
    let signal = TestTensor::<1>::from([1.0, -1.0]);
    let (re, im) = rfft(signal, 0, None);

    let expected_re = TensorData::from([0.0, 2.0]);
    let expected_im = TensorData::from([0.0, 0.0]);

    re.into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-4));
    im.into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-4));
}

#[test]
fn rfft_dim1_sine_wave_produces_imaginary_spectrum() {
    let signal = TestTensor::<2>::from([[
        0.0,
        SQ2INV_PLUS_HALF,
        1.0,
        SQ2INV_MINUS_HALF,
        0.0,
        -SQ2INV_MINUS_HALF,
        -1.0,
        -SQ2INV_PLUS_HALF,
    ]]);
    let dim = 1;
    let (spectrum_re, spectrum_im) = rfft(signal.clone(), dim, None);
    let expected_re = TensorData::from([[0, 0, 0, 0, 0]]);
    let expected_im = TensorData::from([[0, -4, -2, 0, 0]]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), *expected_re.shape());

    spectrum_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-4));

    spectrum_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-4));
}

#[test]
fn rfft_dim1_cosine_wave_produces_real_spectrum() {
    let signal = TestTensor::<2>::from([[1.0, SQ2INV, 0.0, -SQ2INV, -1.0, -SQ2INV, 0.0, SQ2INV]]);

    let (spectrum_re, spectrum_im) = rfft(signal, 1, None);

    let expected_re = TensorData::from([[0.0, 4.0, 0.0, 0.0, 0.0]]);
    let expected_im = TensorData::from([[0.0, 0.0, 0.0, 0.0, 0.0]]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), *expected_re.shape());

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
        [0.0, SQ2INV, 1.0, SQ2INV, 0.0, -SQ2INV, -1.0, -SQ2INV],
        [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
    ]);

    let (spectrum_re, spectrum_im) = rfft(signal, 1, None);

    let expected_re = TensorData::from([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]);

    let expected_im = TensorData::from([[0.0, -4.0, 0.0, 0.0, 0.0], [0.0, 0.0, -4.0, 0.0, 0.0]]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), *expected_re.shape());

    spectrum_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));

    spectrum_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn rfft_negative_dim_matches_positive_dim() {
    let signal = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, -1.0]]);

    let (expected_re, expected_im) = rfft(signal.clone(), 1, None);
    let (actual_re, actual_im) = rfft(signal, -1, None);

    actual_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re.into_data(), Tolerance::absolute(1e-4));
    actual_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im.into_data(), Tolerance::absolute(1e-4));
}

#[test]
fn rfft_dim0_2d_tensor() {
    let signal = TestTensor::<2>::from([
        [0.0, 0.0],
        [SQ2INV, 1.0],
        [1.0, 0.0],
        [SQ2INV, -1.0],
        [0.0, 0.0],
        [-SQ2INV, 1.0],
        [-1.0, 0.0],
        [-SQ2INV, -1.0],
    ]);

    let (spectrum_re, spectrum_im) = rfft(signal, 0, None);

    let expected_re =
        TensorData::from([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);

    let expected_im =
        TensorData::from([[0.0, 0.0], [-4.0, 0.0], [0.0, -4.0], [0.0, 0.0], [0.0, 0.0]]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), *expected_re.shape());

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
            [0.0, SQ2INV, 1.0, SQ2INV, 0.0, -SQ2INV, -1.0, -SQ2INV],
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
        ],
        [
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
            [0.0, SQ2INV, 1.0, SQ2INV, 0.0, -SQ2INV, -1.0, -SQ2INV],
        ],
    ]);

    let (spectrum_re, spectrum_im) = rfft(signal, 2, None);

    let expected_re = TensorData::from([
        [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
    ]);

    let expected_im = TensorData::from([
        [[0.0, -4.0, 0.0, 0.0, 0.0], [0.0, 0.0, -4.0, 0.0, 0.0]],
        [[0.0, 0.0, -4.0, 0.0, 0.0], [0.0, -4.0, 0.0, 0.0, 0.0]],
    ]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), *expected_re.shape());

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

    let (spectrum_re, spectrum_im) = rfft(signal, 1, None);

    let expected_re = TensorData::from([
        [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
    ]);

    let expected_im = TensorData::from([
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ]);

    assert_eq!(spectrum_re.shape(), spectrum_im.shape());
    assert_eq!(spectrum_re.shape(), *expected_re.shape());

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

    let signal = irfft(spectrum_re, spectrum_im, 1, None);

    let expected = TensorData::from([[
        0.0,
        SQ2INV_PLUS_HALF,
        1.0,
        SQ2INV_MINUS_HALF,
        0.0,
        -SQ2INV_MINUS_HALF,
        -1.0,
        -SQ2INV_PLUS_HALF,
    ]]);

    signal
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-3));
}

#[test]
fn irfft_dim1_real_spectrum_produces_cosine_wave() {
    let spectrum_re = TestTensor::<2>::from([[0.0, 4.0, 0.0, 0.0, 0.0]]);
    let spectrum_im = TestTensor::<2>::from([[0.0, 0.0, 0.0, 0.0, 0.0]]);

    let signal = irfft(spectrum_re, spectrum_im, 1, None);

    let expected = TensorData::from([[1.0, SQ2INV, 0.0, -SQ2INV, -1.0, -SQ2INV, 0.0, SQ2INV]]);

    signal
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-3));
}

#[test]
fn irfft_dim1_2d_tensor_distinct_rows() {
    let spectrum_re = TestTensor::<2>::from([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]);

    let spectrum_im =
        TestTensor::<2>::from([[0.0, -4.0, 0.0, 0.0, 0.0], [0.0, 0.0, -4.0, 0.0, 0.0]]);

    let signal = irfft(spectrum_re, spectrum_im, 1, None);

    let expected = TensorData::from([
        [0.0, SQ2INV, 1.0, SQ2INV, 0.0, -SQ2INV, -1.0, -SQ2INV],
        [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
    ]);

    signal
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-3));
}

#[test]
fn irfft_negative_dim_matches_positive_dim() {
    let spectrum_re = TestTensor::<2>::from([[10.0, -2.0, -2.0], [0.0, 0.0, 0.0]]);
    let spectrum_im = TestTensor::<2>::from([[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]]);

    let expected = irfft(spectrum_re.clone(), spectrum_im.clone(), 1, None);
    let actual = irfft(spectrum_re, spectrum_im, -1, None);

    actual
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::absolute(1e-4));
}

#[test]
fn irfft_dim0_2d_tensor() {
    let spectrum_re =
        TestTensor::<2>::from([[0.0, 0.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);

    let spectrum_im =
        TestTensor::<2>::from([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);

    let signal = irfft(spectrum_re, spectrum_im, 0, None);

    let expected = TensorData::from([
        [1.0, 1.0],
        [SQ2INV, SQ2INV],
        [0.0, 0.0],
        [-SQ2INV, -SQ2INV],
        [-1.0, -1.0],
        [-SQ2INV, -SQ2INV],
        [0.0, 0.0],
        [SQ2INV, SQ2INV],
    ]);

    signal
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-3));
}

#[test]
fn rfft_irfft_roundtrip_1d() {
    let signal = TestTensor::<1>::from([
        0.0,
        SQ2INV_PLUS_HALF,
        1.0,
        SQ2INV_MINUS_HALF,
        0.0,
        -SQ2INV_MINUS_HALF,
        -1.0,
        -SQ2INV_PLUS_HALF,
    ]);

    let (re, im) = rfft(signal.clone(), 0, None);
    let reconstructed = irfft(re, im, 0, None);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&signal.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn rfft_irfft_roundtrip_dim1_2d() {
    let signal = TestTensor::<2>::from([
        [0.0, SQ2INV, 1.0, SQ2INV, 0.0, -SQ2INV, -1.0, -SQ2INV],
        [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
    ]);

    let (re, im) = rfft(signal.clone(), 1, None);
    let reconstructed = irfft(re, im, 1, None);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&signal.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn rfft_irfft_roundtrip_dim0_2d() {
    let signal = TestTensor::<2>::from([
        [1.0, 0.0],
        [SQ2INV, 1.0],
        [0.0, 0.0],
        [-SQ2INV, -1.0],
        [-1.0, 0.0],
        [-SQ2INV, 1.0],
        [0.0, 0.0],
        [SQ2INV, -1.0],
    ]);

    let (re, im) = rfft(signal.clone(), 0, None);
    let reconstructed = irfft(re, im, 0, None);

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
            [SQ2INV, 1.0],
            [0.0, 0.0],
            [-SQ2INV, -1.0],
            [-1.0, 0.0],
            [-SQ2INV, 1.0],
            [0.0, 0.0],
            [SQ2INV, -1.0],
        ],
    ]);

    let (re, im) = rfft(signal.clone(), 1, None);
    let reconstructed = irfft(re, im, 1, None);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&signal.into_data(), Tolerance::absolute(1e-3));
}

// ---- Padded input tests (n: Some(...)) ----

#[test]
fn rfft_with_n_larger_than_signal() {
    // Signal of length 4, padded to n=8
    // DFT of [1,0,0,0, 0,0,0,0] = all-ones real, zero imag
    let signal = TestTensor::<1>::from([1.0, 0.0, 0.0, 0.0]);
    let (re, im) = rfft(signal, 0, Some(8));

    let expected_re = TensorData::from([1.0, 1.0, 1.0, 1.0, 1.0]);
    let expected_im = TensorData::from([0.0, 0.0, 0.0, 0.0, 0.0]);

    re.into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));
    im.into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn rfft_with_n_smaller_than_signal() {
    // Signal of length 8, truncated to n=4 -> DFT of [1,0,0,0]
    let signal = TestTensor::<1>::from([1.0, 0.0, 0.0, 0.0, 99.0, 99.0, 99.0, 99.0]);
    let (re, im) = rfft(signal, 0, Some(4));

    let expected_re = TensorData::from([1.0, 1.0, 1.0]);
    let expected_im = TensorData::from([0.0, 0.0, 0.0]);

    re.into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));
    im.into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn rfft_arbitrary_n_batched_dim0_matches_naive() {
    // Non-power-of-two transform along dim 0 of a batched input. The chirp
    // constants are reshaped to `[len, 1, .., 1]`, so a wrong axis only shows
    // up for batched inputs like this one.
    let columns = [
        [1.0f32, 0.0, -1.0, 0.0, 0.5],
        [0.0f32, 1.0, 0.0, -1.0, -0.5],
    ];
    let signal = TestTensor::<2>::from([
        [columns[0][0], columns[1][0]],
        [columns[0][1], columns[1][1]],
        [columns[0][2], columns[1][2]],
        [columns[0][3], columns[1][3]],
        [columns[0][4], columns[1][4]],
    ]);
    let n = 5;
    let (re, im) = rfft(signal, 0, None);
    assert_eq!(re.dims(), [n / 2 + 1, 2]);
    assert_eq!(im.dims(), [n / 2 + 1, 2]);

    for (col, values) in columns.iter().enumerate() {
        let (expected_re, expected_im) = naive_rfft(values, n);
        let actual_re = re
            .clone()
            .narrow(1, col, 1)
            .into_data()
            .try_into_vec::<FloatElem>()
            .unwrap();
        let actual_im = im
            .clone()
            .narrow(1, col, 1)
            .into_data()
            .try_into_vec::<FloatElem>()
            .unwrap();
        for k in 0..(n / 2 + 1) {
            assert!(
                (actual_re[k] as f64 - expected_re[k]).abs() < 1e-3,
                "col={col} k={k} real: got {}, want {}",
                actual_re[k],
                expected_re[k],
            );
            assert!(
                (actual_im[k] as f64 - expected_im[k]).abs() < 1e-3,
                "col={col} k={k} imag: got {}, want {}",
                actual_im[k],
                expected_im[k],
            );
        }
    }
}

#[test]
fn rfft_arbitrary_n_negative_dim_matches_positive_dim() {
    let signal = TestTensor::<2>::from([[1.0, 0.0, -1.0, 0.0, 0.5], [0.0, 1.0, 0.0, -1.0, -0.5]]);

    let (expected_re, expected_im) = rfft(signal.clone(), 1, None);
    let (actual_re, actual_im) = rfft(signal, -1, None);

    actual_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re.into_data(), Tolerance::absolute(1e-4));
    actual_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im.into_data(), Tolerance::absolute(1e-4));
}

/// Direct O(n^2) real DFT used as an independent reference for the Bluestein
/// path. This is the same definition used by `numpy.fft.rfft` and
/// `torch.fft.rfft`, so agreeing with it is what the issue asks for.
///
/// Computed in `f64` with the phase reduced modulo `n`, so the reference stays
/// exact for larger sizes instead of drifting with an f32 angle.
fn naive_rfft(values: &[f32], n: usize) -> (Vec<f64>, Vec<f64>) {
    let bins = n / 2 + 1;
    let mut re = vec![0.0f64; bins];
    let mut im = vec![0.0f64; bins];
    for (k, (re_k, im_k)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
        for t in 0..n {
            let x = values.get(t).copied().unwrap_or(0.0) as f64;
            let angle = -2.0 * core::f64::consts::PI * ((k * t) % n) as f64 / n as f64;
            let (sin, cos) = angle.sin_cos();
            *re_k += x * cos;
            *im_k += x * sin;
        }
    }
    (re, im)
}

fn assert_rfft_matches_naive(values: &[f32], n: usize, tol: f64) {
    let signal = TestTensor::<1>::from(values);
    let (re, im) = rfft(signal, 0, Some(n));
    assert_eq!(re.dims(), [n / 2 + 1]);
    assert_eq!(im.dims(), [n / 2 + 1]);

    let (expected_re, expected_im) = naive_rfft(values, n);
    let re_v = re.into_data().try_into_vec::<FloatElem>().unwrap();
    let im_v = im.into_data().try_into_vec::<FloatElem>().unwrap();
    for k in 0..(n / 2 + 1) {
        assert!(
            (re_v[k] as f64 - expected_re[k]).abs() < tol,
            "n={n} k={k} real: got {}, want {}",
            re_v[k],
            expected_re[k],
        );
        assert!(
            (im_v[k] as f64 - expected_im[k]).abs() < tol,
            "n={n} k={k} imag: got {}, want {}",
            im_v[k],
            expected_im[k],
        );
    }
}

#[test]
fn rfft_arbitrary_n_matches_naive_dft() {
    // Prime and composite non-power-of-two sizes must match the direct DFT
    // definition, not a zero-padded power-of-two grid.
    let values = [
        0.31f32, -0.72, 1.24, 0.53, -0.11, 0.97, -0.44, 0.26, 0.68, -0.83, 0.15, 0.39, -0.58,
    ];
    for &n in &[3usize, 5, 6, 7, 9, 10, 11, 13] {
        assert_rfft_matches_naive(&values[..n], n, 1e-3);
    }
}

#[test]
fn rfft_arbitrary_n_large_matches_naive_dft() {
    // A larger size exercises a long Bluestein convolution; the tolerance is
    // relative to the largest bin magnitude.
    let n = 1000usize;
    let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let (expected_re, expected_im) = naive_rfft(&values, n);
    let max = expected_re
        .iter()
        .chain(expected_im.iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    assert_rfft_matches_naive(&values, n, 1e-3 * max);
}

#[test]
fn rfft_irfft_roundtrip_multiple_arbitrary_n() {
    let values = [1.0f32, -2.0, 3.5, 0.25, -4.0, 1.5, 2.0, -0.5, 0.75];
    for &n in &[3usize, 5, 6, 7, 9] {
        let signal = TestTensor::<1>::from(&values[..n]);
        let (re, im) = rfft(signal.clone(), 0, Some(n));
        let reconstructed = irfft(re, im, 0, Some(n));

        reconstructed
            .into_data()
            .assert_approx_eq::<FloatElem>(&signal.into_data(), Tolerance::absolute(1e-3));
    }
}

/// Direct O(n^2) inverse real DFT matching `numpy.fft.irfft` / `torch.fft.irfft`
/// semantics: the provided half-spectrum is truncated or zero-padded to
/// `n / 2 + 1` bins, mirrored with Hermitian symmetry, then inverted.
fn naive_irfft(re_in: &[f32], im_in: &[f32], n: usize) -> Vec<f64> {
    let half = n / 2 + 1;
    let mut full_re = vec![0.0f64; n];
    let mut full_im = vec![0.0f64; n];
    for k in 0..half {
        let (r, i) = if k < re_in.len() {
            (re_in[k] as f64, im_in[k] as f64)
        } else {
            (0.0, 0.0)
        };
        full_re[k] = r;
        full_im[k] = i;
        if k > 0 && k < n - k {
            full_re[n - k] = r;
            full_im[n - k] = -i;
        }
    }

    let mut out = vec![0.0f64; n];
    for (t, out_t) in out.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for k in 0..n {
            let angle = 2.0 * core::f64::consts::PI * ((k * t) % n) as f64 / n as f64;
            let (sin, cos) = angle.sin_cos();
            acc += full_re[k] * cos - full_im[k] * sin;
        }
        *out_t = acc / n as f64;
    }
    out
}

fn assert_irfft_matches_naive(re_in: &[f32], im_in: &[f32], n: Option<usize>, tol: f64) {
    let out_len = n.unwrap_or((re_in.len() - 1) * 2);
    let re = TestTensor::<1>::from(re_in);
    let im = TestTensor::<1>::from(im_in);
    let signal = irfft(re, im, 0, n);
    assert_eq!(signal.dims(), [out_len]);

    let expected = naive_irfft(re_in, im_in, out_len);
    let actual = signal.into_data().try_into_vec::<FloatElem>().unwrap();
    for t in 0..out_len {
        assert!(
            (actual[t] as f64 - expected[t]).abs() < tol,
            "n={out_len} t={t}: got {}, want {}",
            actual[t],
            expected[t],
        );
    }
}

#[test]
fn irfft_none_resolves_to_non_power_of_two() {
    // 4 bins with `n = None` reconstruct a length `2 * (4 - 1) = 6` signal,
    // which is not a power of two and therefore uses Bluestein.
    let re = [0.0f32, 1.0, 0.5, -0.25];
    let im = [0.0f32, -1.0, 0.5, 0.25];
    assert_irfft_matches_naive(&re, &im, None, 1e-3);
}

#[test]
fn irfft_arbitrary_n_bin_count_truncate_and_pad() {
    let re = [0.0f32, 1.0, 0.5, -0.25, 0.75];
    let im = [0.0f32, -1.0, 0.5, 0.25, -0.5];
    // n=5: fewer output bins than provided -> truncate.
    // n=9: output bins match the provided count.
    // n=13: more output bins than provided -> zero-pad.
    for &n in &[5usize, 9, 13] {
        assert_irfft_matches_naive(&re, &im, Some(n), 1e-3);
    }
}

#[test]
fn rfft_irfft_roundtrip_with_n() {
    let signal = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]);
    let (re, im) = rfft(signal.clone(), 0, Some(4));
    let reconstructed = irfft(re, im, 0, Some(4));

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&signal.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn irfft_with_n_different_from_natural() {
    // Spectrum from length-4 signal (3 bins), reconstruct at length 8
    let signal = TestTensor::<1>::from([1.0, 0.0, 0.0, 0.0]);
    let (re, im) = rfft(signal, 0, None);
    let reconstructed = irfft(re, im, 0, Some(8));
    assert_eq!(reconstructed.dims(), [8]);
}

#[test]
fn rfft_2d_with_n_padded() {
    // 2D tensor, rfft along dim=1 with n=8 (signal is length 4)
    let signal = TestTensor::<2>::from([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]);
    let (re, im) = rfft(signal, 1, Some(8));

    // Output: 8/2+1=5 frequency bins
    assert_eq!(re.dims(), [2, 5]);
    assert_eq!(im.dims(), [2, 5]);

    // Row 0: impulse zero-padded to 8 -> all-ones real
    let re_data = re.into_data();
    let re_vals = re_data.try_to_vec::<f32>().unwrap();
    for (k, value) in re_vals.iter().take(5).enumerate() {
        assert!(
            (value - 1.0).abs() < 1e-3,
            "row0 re[{k}] should be 1.0, got {value}",
        );
    }
}

// ---- cfft tests ----

#[test]
fn cfft_output_has_n_bins() {
    // cfft should return N bins, not N/2+1
    let re = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]);
    let im = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0]);
    let (out_re, out_im) = cfft(re, im, 0, None);

    assert_eq!(out_re.dims(), [4]);
    assert_eq!(out_im.dims(), [4]);
}

#[test]
fn cfft_pure_real_input() {
    // When imaginary part is zero, cfft should produce the same result
    // as extending rfft to the full spectrum
    let signal = [1.0f32, 2.0, 3.0, 4.0];
    let re = TestTensor::<1>::from(signal);
    let im = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0]);

    let (cfft_re, cfft_im) = cfft(re, im, 0, None);

    // Expected: DFT of [1,2,3,4]
    // X[0] = 10, X[1] = -2+2i, X[2] = -2, X[3] = -2-2i
    let expected_re = TensorData::from([10.0, -2.0, -2.0, -2.0]);
    let expected_im = TensorData::from([0.0, 2.0, 0.0, -2.0]);

    cfft_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));
    cfft_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn cfft_pure_imaginary_input() {
    // Signal is purely imaginary: z[n] = i * [1, 2, 3, 4]
    // FFT(i*x) = i*FFT(x), so result_re = -FFT(x)_im, result_im = FFT(x)_re
    let re = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0]);
    let im = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]);

    let (cfft_re, cfft_im) = cfft(re, im, 0, None);

    // FFT([1,2,3,4]) = [10, -2+2i, -2, -2-2i]
    // i * FFT(x) = i * [10, -2+2i, -2, -2-2i]
    //            = [-0, -2+(-2)i, 0, 2+(-2)i]  → re = [0, -2, 0, 2], im = [10, -2, -2, -2]
    let expected_re = TensorData::from([0.0, -2.0, 0.0, 2.0]);
    let expected_im = TensorData::from([10.0, -2.0, -2.0, -2.0]);

    cfft_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));
    cfft_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn cfft_complex_exponential() {
    // z[n] = exp(i * 2π * n / 4) for n=0..3, i.e. frequency bin 1
    // re = [cos(0), cos(π/2), cos(π), cos(3π/2)] = [1, 0, -1, 0]
    // im = [sin(0), sin(π/2), sin(π), sin(3π/2)] = [0, 1, 0, -1]
    // DFT should be: X[0]=0, X[1]=4, X[2]=0, X[3]=0
    let re = TestTensor::<1>::from([1.0, 0.0, -1.0, 0.0]);
    let im = TestTensor::<1>::from([0.0, 1.0, 0.0, -1.0]);

    let (cfft_re, cfft_im) = cfft(re, im, 0, None);

    let expected_re = TensorData::from([0.0, 4.0, 0.0, 0.0]);
    let expected_im = TensorData::from([0.0, 0.0, 0.0, 0.0]);

    cfft_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));
    cfft_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn cfft_zeros() {
    let re = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0]);
    let im = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0]);

    let (cfft_re, cfft_im) = cfft(re, im, 0, None);

    let expected = TensorData::from([0.0, 0.0, 0.0, 0.0]);

    cfft_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-4));
    cfft_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-4));
}

#[test]
fn cfft_dim1_2d_tensor() {
    // Apply cfft along dim=1 on a 2D tensor
    // Row 0: pure real [1, 2, 3, 4] → DFT = [10, -2+2i, -2, -2-2i]
    // Row 1: complex exponential exp(i·2π·n/4) → DFT = [0, 4, 0, 0]
    let re = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0], [1.0, 0.0, -1.0, 0.0]]);
    let im = TestTensor::<2>::from([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0]]);

    let (cfft_re, cfft_im) = cfft(re, im, 1, None);

    // Output should be [2, 4] (N=4 bins per row)
    assert_eq!(cfft_re.dims(), [2, 4]);
    assert_eq!(cfft_im.dims(), [2, 4]);

    // Row 0: DFT of [1,2,3,4]+i*0 = [10, -2+2i, -2, -2-2i]
    // Row 1: DFT of exp(i*2π*n/4) = [0, 4, 0, 0]
    let expected_re = TensorData::from([[10.0, -2.0, -2.0, -2.0], [0.0, 4.0, 0.0, 0.0]]);
    let expected_im = TensorData::from([[0.0, 2.0, 0.0, -2.0], [0.0, 0.0, 0.0, 0.0]]);

    cfft_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));
    cfft_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn cfft_negative_dim_matches_positive_dim() {
    let re = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0], [1.0, 0.0, -1.0, 0.0]]);
    let im = TestTensor::<2>::from([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0]]);

    let (expected_re, expected_im) = cfft(re.clone(), im.clone(), 1, None);
    let (actual_re, actual_im) = cfft(re, im, -1, None);

    actual_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re.into_data(), Tolerance::absolute(1e-4));
    actual_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im.into_data(), Tolerance::absolute(1e-4));
}

#[test]
fn cfft_with_n_padding() {
    // Signal length 2, padded to N=4
    // z = [1+0i, 0+0i] padded to [1+0i, 0, 0, 0]
    // DFT = [1, 1, 1, 1] (all real, zero imag)
    let re = TestTensor::<1>::from([1.0, 0.0]);
    let im = TestTensor::<1>::from([0.0, 0.0]);

    let (cfft_re, cfft_im) = cfft(re, im, 0, Some(4));

    assert_eq!(cfft_re.dims(), [4]);

    let expected_re = TensorData::from([1.0, 1.0, 1.0, 1.0]);
    let expected_im = TensorData::from([0.0, 0.0, 0.0, 0.0]);

    cfft_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));
    cfft_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn cfft_length_1() {
    assert_length_one_behavior(|| {
        // N=1: DFT of a single complex value is itself
        let re = TestTensor::<1>::from([3.0]);
        let im = TestTensor::<1>::from([5.0]);

        let (cfft_re, cfft_im) = cfft(re, im, 0, None);

        assert_eq!(cfft_re.dims(), [1]);
        cfft_re
            .into_data()
            .assert_approx_eq::<FloatElem>(&TensorData::from([3.0]), Tolerance::absolute(1e-4));
        cfft_im
            .into_data()
            .assert_approx_eq::<FloatElem>(&TensorData::from([5.0]), Tolerance::absolute(1e-4));
    });
}

#[test]
fn cfft_length_2() {
    // N=2: z = [a, b] → X[0] = a+b, X[1] = a-b
    // z = [1+2i, 3+4i]
    // X[0] = (1+3) + i(2+4) = 4+6i
    // X[1] = (1-3) + i(2-4) = -2-2i
    let re = TestTensor::<1>::from([1.0, 3.0]);
    let im = TestTensor::<1>::from([2.0, 4.0]);

    let (cfft_re, cfft_im) = cfft(re, im, 0, None);

    assert_eq!(cfft_re.dims(), [2]);
    cfft_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([4.0, -2.0]), Tolerance::absolute(1e-4));
    cfft_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([6.0, -2.0]), Tolerance::absolute(1e-4));
}

#[test]
#[should_panic(expected = "same shape")]
fn cfft_rejects_mismatched_shapes() {
    let re = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]);
    let im = TestTensor::<1>::from([1.0, 2.0]);
    let _ = cfft(re, im, 0, None);
}

#[test]
#[should_panic(expected = "same shape")]
fn irfft_rejects_mismatched_shapes() {
    // Also validates the non-power-of-two path, which otherwise broadcasts.
    let re = TestTensor::<1>::from([1.0, 2.0, 3.0]);
    let im = TestTensor::<1>::from([1.0, 2.0]);
    let _ = irfft(re, im, 0, Some(5));
}

#[test]
fn cfft_dim0_2d_tensor() {
    // Apply cfft along dim=0 on a 2D tensor (4 rows, 2 columns)
    // Column 0: complex exponential exp(i·2π·n/4) → DFT = [0, 4, 0, 0]
    // Column 1: pure real [1, 2, 3, 4] → DFT = [10, -2+2i, -2, -2-2i]
    let re = TestTensor::<2>::from([[1.0, 1.0], [0.0, 2.0], [-1.0, 3.0], [0.0, 4.0]]);
    let im = TestTensor::<2>::from([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]]);

    let (cfft_re, cfft_im) = cfft(re, im, 0, None);

    assert_eq!(cfft_re.dims(), [4, 2]);
    assert_eq!(cfft_im.dims(), [4, 2]);

    let expected_re = TensorData::from([[0.0, 10.0], [4.0, -2.0], [0.0, -2.0], [0.0, -2.0]]);
    let expected_im = TensorData::from([[0.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, -2.0]]);

    cfft_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));
    cfft_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn cfft_with_n_truncation() {
    // Signal length 8, truncated to n=4 → DFT of [1+0i, 2+0i, 3+0i, 4+0i]
    // Trailing values are discarded, not included in the transform.
    let re = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0, 99.0, 99.0, 99.0, 99.0]);
    let im = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0, 99.0, 99.0, 99.0, 99.0]);

    let (cfft_re, cfft_im) = cfft(re, im, 0, Some(4));

    assert_eq!(cfft_re.dims(), [4]);

    // DFT of [1,2,3,4] = [10, -2+2i, -2, -2-2i]
    let expected_re = TensorData::from([10.0, -2.0, -2.0, -2.0]);
    let expected_im = TensorData::from([0.0, 2.0, 0.0, -2.0]);

    cfft_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));
    cfft_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

// ---- ifft tests ----

#[test]
fn ifft_inverts_cfft() {
    // ifft(cfft(z)) == z for a complex signal.
    let re = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]);
    let im = TestTensor::<1>::from([0.5, -1.0, 0.25, 2.0]);
    let (spec_re, spec_im) = cfft(re.clone(), im.clone(), 0, None);
    let (out_re, out_im) = ifft(spec_re, spec_im, 0, None);

    out_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&re.into_data(), Tolerance::absolute(1e-3));
    out_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&im.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn ifft_impulse_spectrum() {
    // X = [1, 0, 0, 0] → x[n] = 1/N for all n.
    let re = TestTensor::<1>::from([1.0, 0.0, 0.0, 0.0]);
    let im = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0]);
    let (out_re, out_im) = ifft(re, im, 0, None);

    out_re.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([0.25, 0.25, 0.25, 0.25]),
        Tolerance::absolute(1e-4),
    );
    out_im.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([0.0, 0.0, 0.0, 0.0]),
        Tolerance::absolute(1e-4),
    );
}

#[test]
fn ifft_single_bin_complex_exponential() {
    // X = [0, 4, 0, 0] → x[n] = exp(i·2π·n/4) = [1, i, -1, -i].
    let re = TestTensor::<1>::from([0.0, 4.0, 0.0, 0.0]);
    let im = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0]);
    let (out_re, out_im) = ifft(re, im, 0, None);

    out_re.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([1.0, 0.0, -1.0, 0.0]),
        Tolerance::absolute(1e-3),
    );
    out_im.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([0.0, 1.0, 0.0, -1.0]),
        Tolerance::absolute(1e-3),
    );
}

#[test]
fn ifft_dim1_2d_tensor() {
    // Inverse of the spectra in `cfft_dim1_2d_tensor`.
    let spec_re = TestTensor::<2>::from([[10.0, -2.0, -2.0, -2.0], [0.0, 4.0, 0.0, 0.0]]);
    let spec_im = TestTensor::<2>::from([[0.0, 2.0, 0.0, -2.0], [0.0, 0.0, 0.0, 0.0]]);
    let (out_re, out_im) = ifft(spec_re, spec_im, 1, None);

    assert_eq!(out_re.dims(), [2, 4]);
    assert_eq!(out_im.dims(), [2, 4]);

    let expected_re = TensorData::from([[1.0, 2.0, 3.0, 4.0], [1.0, 0.0, -1.0, 0.0]]);
    let expected_im = TensorData::from([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0]]);

    out_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re, Tolerance::absolute(1e-3));
    out_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im, Tolerance::absolute(1e-3));
}

#[test]
fn ifft_negative_dim_matches_positive_dim() {
    let re = TestTensor::<2>::from([[10.0, -2.0, -2.0, -2.0], [0.0, 4.0, 0.0, 0.0]]);
    let im = TestTensor::<2>::from([[0.0, 2.0, 0.0, -2.0], [0.0, 0.0, 0.0, 0.0]]);

    let (expected_re, expected_im) = ifft(re.clone(), im.clone(), 1, None);
    let (actual_re, actual_im) = ifft(re, im, -1, None);

    actual_re
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_re.into_data(), Tolerance::absolute(1e-4));
    actual_im
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_im.into_data(), Tolerance::absolute(1e-4));
}

#[test]
fn ifft_with_n_padding() {
    // X = [4, 0] padded to N=4 → x[n] = 1 for all n.
    let re = TestTensor::<1>::from([4.0, 0.0]);
    let im = TestTensor::<1>::from([0.0, 0.0]);
    let (out_re, out_im) = ifft(re, im, 0, Some(4));

    assert_eq!(out_re.dims(), [4]);
    out_re.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([1.0, 1.0, 1.0, 1.0]),
        Tolerance::absolute(1e-4),
    );
    out_im.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([0.0, 0.0, 0.0, 0.0]),
        Tolerance::absolute(1e-4),
    );
}

#[test]
fn ifft_with_n_truncation_resizes_round_trip() {
    // Round trip at n=4 truncates the length-8 signal, so only its first four
    // samples are recovered. The output has length n, not the input length.
    let re = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let im = TestTensor::<1>::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let (spec_re, spec_im) = cfft(re, im, 0, Some(4));
    assert_eq!(spec_re.dims(), [4]);

    let (out_re, out_im) = ifft(spec_re, spec_im, 0, Some(4));
    assert_eq!(out_re.dims(), [4]);

    out_re.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([1.0, 2.0, 3.0, 4.0]),
        Tolerance::absolute(1e-3),
    );
    out_im.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([0.0, 0.0, 0.0, 0.0]),
        Tolerance::absolute(1e-3),
    );
}

#[test]
fn ifft_length_1() {
    assert_length_one_behavior(|| {
        // N=1: the inverse of a single complex value is itself.
        let re = TestTensor::<1>::from([3.0]);
        let im = TestTensor::<1>::from([5.0]);
        let (out_re, out_im) = ifft(re, im, 0, None);

        out_re
            .into_data()
            .assert_approx_eq::<FloatElem>(&TensorData::from([3.0]), Tolerance::absolute(1e-4));
        out_im
            .into_data()
            .assert_approx_eq::<FloatElem>(&TensorData::from([5.0]), Tolerance::absolute(1e-4));
    });
}

#[test]
#[should_panic(expected = "same shape")]
fn ifft_rejects_mismatched_shapes() {
    let re = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]);
    let im = TestTensor::<1>::from([1.0, 2.0]);
    let _ = ifft(re, im, 0, None);
}

/// Direct O(n^2) complex DFT, used as an independent reference for `cfft` at
/// non-power-of-two sizes.
fn naive_cfft(re_in: &[f32], im_in: &[f32], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    for (k, (re_k, im_k)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
        for t in 0..n {
            let xr = re_in.get(t).copied().unwrap_or(0.0) as f64;
            let xi = im_in.get(t).copied().unwrap_or(0.0) as f64;
            let angle = -2.0 * core::f64::consts::PI * ((k * t) % n) as f64 / n as f64;
            let (sin, cos) = angle.sin_cos();
            *re_k += xr * cos - xi * sin;
            *im_k += xr * sin + xi * cos;
        }
    }
    (re, im)
}

#[test]
fn cfft_arbitrary_n_matches_naive_dft() {
    let re_in = [0.31f32, -0.72, 1.24, 0.53, -0.11, 0.97, -0.44];
    let im_in = [-0.2f32, 0.5, 0.1, -0.3, 0.8, -0.6, 0.4];
    for &n in &[3usize, 5, 6, 7] {
        let re = TestTensor::<1>::from(&re_in[..n]);
        let im = TestTensor::<1>::from(&im_in[..n]);
        let (out_re, out_im) = cfft(re, im, 0, Some(n));
        assert_eq!(out_re.dims(), [n]);
        assert_eq!(out_im.dims(), [n]);

        let (expected_re, expected_im) = naive_cfft(&re_in[..n], &im_in[..n], n);
        let actual_re = out_re.into_data().try_into_vec::<FloatElem>().unwrap();
        let actual_im = out_im.into_data().try_into_vec::<FloatElem>().unwrap();
        for k in 0..n {
            assert!(
                (actual_re[k] as f64 - expected_re[k]).abs() < 1e-3,
                "n={n} k={k} real: got {}, want {}",
                actual_re[k],
                expected_re[k],
            );
            assert!(
                (actual_im[k] as f64 - expected_im[k]).abs() < 1e-3,
                "n={n} k={k} imag: got {}, want {}",
                actual_im[k],
                expected_im[k],
            );
        }
    }
}

// CubeCL currently rejects length-one FFTs; Flex supports them.
fn assert_length_one_behavior(run: impl FnOnce() + std::panic::UnwindSafe) {
    #[cfg(feature = "cubecl-backend")]
    if matches!(
        burn_core::tensor::Device::default().as_dispatch(),
        burn_core::backend::DispatchDevice::Cube(_)
    ) {
        assert!(std::panic::catch_unwind(run).is_err());
        return;
    }
    run();
}

// Quantization is implemented by the Flex and CubeCL backends, but not by tch.
#[cfg(any(feature = "flex", feature = "cubecl-backend"))]
#[test]
fn rfft_quantized_non_power_of_two_matches_dequantized() {
    use burn_core::tensor::quantization::{QuantStore, QuantValue};

    let device = burn_core::tensor::Device::default();
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_store(QuantStore::Native);

    let signal = TestTensor::<1>::from([0.5, -1.0, 2.0, -0.25, 0.75, 1.5]);
    let quantized = signal.clone().quantize_dynamic(&scheme);
    let dequantized = quantized.clone().dequantize();

    // n=5 is not a power of two, so this used to panic inside `zeros_like`.
    let (re_q, im_q) = rfft(quantized, 0, Some(5));
    let (re_d, im_d) = rfft(dequantized, 0, Some(5));

    re_q.into_data()
        .assert_approx_eq::<FloatElem>(&re_d.into_data(), Tolerance::absolute(1e-4));
    im_q.into_data()
        .assert_approx_eq::<FloatElem>(&im_d.into_data(), Tolerance::absolute(1e-4));
}
