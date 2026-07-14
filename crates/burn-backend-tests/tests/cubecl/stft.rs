use super::*;
use burn_tensor::Tolerance;
use burn_tensor::signal::{StftOptions, hann_window, istft, stft};

fn opts(n_fft: usize, hop_length: usize, center: bool, onesided: bool) -> StftOptions {
    StftOptions {
        n_fft,
        hop_length,
        win_length: None,
        center,
        onesided,
    }
}

#[test]
fn stft_constant_signal_rectangular_window() {
    // Constant signal with rectangular window: only DC bin should be non-zero
    let signal = TestTensor::<2>::from([[1.0, 1.0, 1.0, 1.0]]);
    let result = stft(signal, None, opts(4, 1, false, true));

    let [batch, n_frames, n_freqs, two] = result.dims();
    assert_eq!(batch, 1);
    assert_eq!(n_frames, 1); // (4 - 4) / 1 + 1 = 1
    assert_eq!(n_freqs, 3); // 4/2 + 1 = 3
    assert_eq!(two, 2);

    // DC bin should be 4.0 + 0i (sum of ones)
    let data = result.into_data();
    let values = data.to_vec::<f32>().unwrap();
    // [batch=0, frame=0, freq=0, re] = 4.0
    assert!(
        (values[0] - 4.0).abs() < 1e-4,
        "DC real should be 4.0, got {}",
        values[0]
    );
    // [batch=0, frame=0, freq=0, im] = 0.0
    assert!(
        values[1].abs() < 1e-4,
        "DC imag should be 0.0, got {}",
        values[1]
    );
}

#[test]
fn stft_output_shape_onesided() {
    let signal = TestTensor::<2>::from([[1.0; 16]]);
    let result = stft(signal, None, opts(8, 4, false, true));

    let [batch, n_frames, n_freqs, two] = result.dims();
    assert_eq!(batch, 1);
    assert_eq!(n_frames, 3); // (16 - 8) / 4 + 1 = 3
    assert_eq!(n_freqs, 5); // 8/2 + 1 = 5
    assert_eq!(two, 2);
}

#[test]
fn stft_output_shape_twosided() {
    let signal = TestTensor::<2>::from([[1.0; 16]]);
    let result = stft(signal, None, opts(8, 4, false, false));

    let [batch, n_frames, n_freqs, two] = result.dims();
    assert_eq!(batch, 1);
    assert_eq!(n_frames, 3);
    assert_eq!(n_freqs, 8); // full spectrum
    assert_eq!(two, 2);
}

#[test]
fn stft_center_padding() {
    // With center=true, signal is padded by n_fft/2 on both sides
    let signal = TestTensor::<2>::from([[1.0; 8]]);
    let result = stft(signal, None, opts(4, 2, true, true));

    // After padding: 2 + 8 + 2 = 12 samples
    // n_frames = (12 - 4) / 2 + 1 = 5
    let [_, n_frames, _, _] = result.dims();
    assert_eq!(n_frames, 5);
}

#[test]
fn stft_with_hann_window() {
    let signal = TestTensor::<2>::from([[1.0; 8]]);
    let window: TestTensor<1> = hann_window(4, true, &Default::default());
    let result = stft(signal, Some(window), opts(4, 2, false, true));

    let [batch, n_frames, n_freqs, two] = result.dims();
    assert_eq!(batch, 1);
    assert_eq!(n_frames, 3);
    assert_eq!(n_freqs, 3);
    assert_eq!(two, 2);
}

#[test]
fn stft_batch_dimension() {
    let signal = TestTensor::<2>::from([[1.0; 8], [2.0; 8]]);
    let result = stft(signal, None, opts(4, 2, false, true));

    let [batch, _, _, _] = result.dims();
    assert_eq!(batch, 2);
}

#[test]
fn stft_istft_roundtrip_rectangular() {
    let original = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
    let n_fft = 4;
    let hop_length = 2;

    let o = opts(n_fft, hop_length, false, true);
    let spectrum = stft(original.clone(), None, o);
    let reconstructed = istft(spectrum, None, Some(8), o);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&original.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn stft_istft_roundtrip_centered() {
    let original = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
    let n_fft = 4;
    let hop_length = 2;

    let o = opts(n_fft, hop_length, true, true);
    let spectrum = stft(original.clone(), None, o);
    let reconstructed = istft(spectrum, None, Some(8), o);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&original.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn stft_istft_roundtrip_twosided() {
    let original = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
    let n_fft = 4;
    let hop_length = 2;

    let o = opts(n_fft, hop_length, false, false);
    let spectrum = stft(original.clone(), None, o);
    let reconstructed = istft(spectrum, None, Some(8), o);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&original.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn stft_istft_roundtrip_batch() {
    let original = TestTensor::<2>::from([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ]);
    let n_fft = 4;
    let hop_length = 2;

    let o = opts(n_fft, hop_length, false, true);
    let spectrum = stft(original.clone(), None, o);
    let reconstructed = istft(spectrum, None, Some(8), o);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&original.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn stft_istft_roundtrip_hann_window() {
    let original = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
    let n_fft = 4;
    let hop_length = 1;

    let window: TestTensor<1> = hann_window(4, true, &Default::default());
    let o = opts(n_fft, hop_length, true, true);
    let spectrum = stft(original.clone(), Some(window.clone()), o);
    let reconstructed = istft(spectrum, Some(window), Some(8), o);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&original.into_data(), Tolerance::absolute(1e-2));
}

#[test]
fn stft_with_hamming_window() {
    use burn_tensor::signal::hamming_window;
    let signal = TestTensor::<2>::from([[1.0; 8]]);
    let window: TestTensor<1> = hamming_window(4, true, &Default::default());
    let result = stft(signal, Some(window), opts(4, 2, false, true));

    let [batch, n_frames, n_freqs, two] = result.dims();
    assert_eq!(batch, 1);
    assert_eq!(n_frames, 3);
    assert_eq!(n_freqs, 3);
    assert_eq!(two, 2);
}

#[test]
#[should_panic(expected = "hop_length")]
fn stft_rejects_hop_greater_than_window() {
    // hop (5) > effective win_length (4) violates COLA/NOLA; must be rejected.
    let signal = TestTensor::<2>::from([[1.0; 16]]);
    let _ = stft(signal, None, opts(4, 5, false, true));
}

#[test]
#[should_panic(expected = "n_fft")]
fn stft_rejects_zero_nfft() {
    let signal = TestTensor::<2>::from([[1.0; 4]]);
    let _ = stft(signal, None, opts(0, 1, false, true));
}

#[test]
#[should_panic(expected = "power of two")]
fn stft_rejects_non_power_of_two_nfft() {
    // n_fft=5 is not a power of two; should hard-fail in StftOptions::assert_valid.
    let signal = TestTensor::<2>::from([[1.0; 8]]);
    let _ = stft(signal, None, opts(5, 1, false, true));
}

#[test]
#[should_panic(expected = "hop_length")]
fn stft_rejects_zero_hop_length() {
    let signal = TestTensor::<2>::from([[1.0; 4]]);
    let _ = stft(signal, None, opts(4, 0, false, true));
}

#[test]
#[should_panic(expected = "win_length")]
fn stft_rejects_zero_win_length() {
    let signal = TestTensor::<2>::from([[1.0; 4]]);
    let o = StftOptions {
        n_fft: 4,
        hop_length: 1,
        win_length: Some(0),
        center: false,
        onesided: true,
    };
    let _ = stft(signal, None, o);
}

#[test]
#[should_panic(expected = "reflect pad")]
fn stft_rejects_too_short_signal_with_center() {
    // n_fft/2 = 2, signal length 2 is not > 2, so reflect pad would fail.
    let signal = TestTensor::<2>::from([[1.0, 2.0]]);
    let _ = stft(signal, None, opts(4, 1, true, true));
}

#[test]
#[should_panic(expected = "window length")]
fn istft_rejects_wrong_window_length() {
    // Synthetic stft matrix: [batch=1, n_frames=3, n_freqs=3, 2 (re/im)].
    // Values are arbitrary; we only need istft to reach the window-length check.
    let spectrum: TestTensor<4> = TestTensor::from([[
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ]]);

    // n_fft=4, effective win_length=3 (per win_length=Some(3)); passed window length=4 mismatches.
    let bad_window: TestTensor<1> = TestTensor::from([1.0, 1.0, 1.0, 1.0]);
    let o_bad = StftOptions {
        n_fft: 4,
        hop_length: 2,
        win_length: Some(3),
        center: false,
        onesided: true,
    };
    let _ = istft(spectrum, Some(bad_window), Some(8), o_bad);
}

#[test]
#[should_panic(expected = "n_freqs")]
fn istft_rejects_wrong_n_freqs() {
    // n_fft=4, onesided=true: expected n_freqs = 4/2+1 = 3.
    // Pass a spectrum with 4 bins to trigger the shape check.
    let spectrum: TestTensor<4> = TestTensor::from([[
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ]]);
    let _ = istft(spectrum, None, Some(8), opts(4, 2, false, true));
}

#[test]
fn stft_options_default_and_new() {
    // Spot-check the defaults match PyTorch (hop = n_fft/4, center, onesided).
    let o = StftOptions::new(16);
    assert_eq!(o.n_fft, 16);
    assert_eq!(o.hop_length, 4);
    assert_eq!(o.win_length, None);
    assert!(o.center);
    assert!(o.onesided);
    let d = StftOptions::default();
    assert_eq!(d.n_fft, 400);
}
