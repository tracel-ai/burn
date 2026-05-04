use super::*;
use burn_tensor::signal::{cfft, irfft, rfft};
use burn_tensor::{TensorData, Tolerance};

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
    let signal = TestTensor::<1>::from([5.0]);
    let (re, im) = rfft(signal, 0, None);

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
    let signal = TestTensor::<2>::from([[0.0, 1.2071, 1.0, 0.2071, 0.0, -0.2071, -1.0, -1.2071]]);
    let dim = 1;
    let (spectrum_re, spectrum_im) = rfft(signal.clone(), dim, None);
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

    let (spectrum_re, spectrum_im) = rfft(signal, 1, None);

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

    let (spectrum_re, spectrum_im) = rfft(signal, 1, None);

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

    let (spectrum_re, spectrum_im) = rfft(signal, 0, None);

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

    let signal = irfft(spectrum_re, spectrum_im, 1, None);

    let expected = TensorData::from([[0.0, 1.2071, 1.0, 0.2071, 0.0, -0.2071, -1.0, -1.2071]]);

    signal
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-3));
}

#[test]
fn irfft_dim1_real_spectrum_produces_cosine_wave() {
    let spectrum_re = TestTensor::<2>::from([[0.0, 4.0, 0.0, 0.0, 0.0]]);
    let spectrum_im = TestTensor::<2>::from([[0.0, 0.0, 0.0, 0.0, 0.0]]);

    let signal = irfft(spectrum_re, spectrum_im, 1, None);

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

    let signal = irfft(spectrum_re, spectrum_im, 1, None);

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

    let signal = irfft(spectrum_re, spectrum_im, 0, None);

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

    let (re, im) = rfft(signal.clone(), 0, None);
    let reconstructed = irfft(re, im, 0, None);

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
        [0.7071, 1.0],
        [0.0, 0.0],
        [-0.7071, -1.0],
        [-1.0, 0.0],
        [-0.7071, 1.0],
        [0.0, 0.0],
        [0.7071, -1.0],
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
            [0.7071, 1.0],
            [0.0, 0.0],
            [-0.7071, -1.0],
            [-1.0, 0.0],
            [-0.7071, 1.0],
            [0.0, 0.0],
            [0.7071, -1.0],
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
#[should_panic(expected = "power of two")]
fn rfft_rejects_non_power_of_two_n() {
    let signal = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]);
    let _ = rfft(signal, 0, Some(5));
}

#[test]
#[should_panic(expected = "power of two")]
fn irfft_rejects_non_power_of_two_n() {
    let re = TestTensor::<1>::from([1.0, 2.0, 3.0]);
    let im = TestTensor::<1>::from([0.0, 0.0, 0.0]);
    let _ = irfft(re, im, 0, Some(5));
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
    let re_vals = re_data.to_vec::<f32>().unwrap();
    for k in 0..5 {
        assert!(
            (re_vals[k] - 1.0).abs() < 1e-3,
            "row0 re[{k}] should be 1.0, got {}",
            re_vals[k]
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
