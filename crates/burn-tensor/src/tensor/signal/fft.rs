use burn_backend::Backend;

use crate::Tensor;
use crate::TensorPrimitive;
use crate::check;
use crate::check::TensorCheck;

/// Computes the 1-dimensional discrete Fourier Transform of real-valued input.
///
/// Since the input is real, the Hermitian symmetry is exploited, and only the
/// first non-redundant values are returned ($N/2 + 1$).
/// For now, the autodiff is not yet supported
///
#[cfg_attr(
    doc,
    doc = r#"
The mathematical formulation for each element $k$ in the frequency domain is:

$$X\[k\] = \sum_{n=0}^{N-1} x\[n\] \left\[ \cos\left(\frac{2\pi kn}{N}\right) - i \sin\left(\frac{2\pi kn}{N}\right) \right\]$$

where $N$ is the size of the signal along the specified dimension.
"#
)]
#[cfg_attr(not(doc), doc = r"X\[k\] = Σ x\[n\] * exp(-i*2πkn/N)")]
///
/// # Arguments
///
/// * `signal` - The input tensor containing the real-valued signal.
/// * `dim` - The dimension along which to take the FFT.
/// * `n` - Optional FFT length. When `None`, the signal must be a power of two along `dim`.
///   When `Some(n)`, `n` must also be a power of two; the signal is truncated or zero-padded
///   to length `n`. Non-power-of-two `n` is rejected with a panic (true arbitrary-size DFT
///   support via Bluestein's algorithm is tracked as a follow-up).
///
/// # Returns
///
/// A tuple containing:
/// 1. The real part of the spectrum. Output length along `dim` is `n / 2 + 1` (using `n` or
///    `signal_len` respectively).
/// 2. The imaginary part of the spectrum (same shape).
///
/// # Example
///
/// ```rust
/// use burn_tensor::backend::Backend;
/// use burn_tensor::Tensor;
///
/// fn example<B: Backend>() {
///     let device = B::Device::default();
///     let signal = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
///     let (real, imag) = burn_tensor::signal::rfft(signal, 0, None);
/// }
/// ```
pub fn rfft<B: Backend, const D: usize>(
    signal: Tensor<B, D>,
    dim: usize,
    n: Option<usize>,
) -> (Tensor<B, D>, Tensor<B, D>) {
    check!(TensorCheck::check_dim::<D>(dim));

    match n {
        None => check!(TensorCheck::check_is_power_of_two::<D>(
            &signal.shape(),
            dim
        )),
        Some(n) => {
            assert!(n >= 1, "rfft: n must be >= 1, got {n}");
            assert!(
                n.is_power_of_two(),
                "rfft: n must be a power of two, got {n}. True non-power-of-two \
                 DFT support is tracked as a follow-up (Bluestein's algorithm)."
            );
        }
    }

    let (spectrum_re, spectrum_im) = B::rfft(signal.primitive.tensor(), dim, n);
    (
        Tensor::new(TensorPrimitive::Float(spectrum_re)),
        Tensor::new(TensorPrimitive::Float(spectrum_im)),
    )
}

/// Computes the 1-dimensional inverse discrete Fourier Transform for real-valued signals.
///
/// This function reconstructs the real-valued time-domain signal from the
/// first non-redundant values ($N/2 + 1$) of the frequency-domain spectrum.
/// For now, the autodiff is not yet supported.
///
#[cfg_attr(
    doc,
    doc = r#"
The mathematical formulation for each element $n$ in the time domain is:

$$x\[n\] = \frac{1}{N} \sum_{k=0}^{N-1} X\[k\] \left\[ \cos\left(\frac{2\pi kn}{N}\right) + i \sin\left(\frac{2\pi kn}{N}\right) \right\]$$

where $N$ is the size of the reconstructed signal.
"#
)]
#[cfg_attr(not(doc), doc = r"x\[n\] = (1/N) * Σ X\[k\] * exp(i*2πkn/N)")]
///
/// # Arguments
///
/// * `spectrum_re` - The real part of the spectrum.
/// * `spectrum_im` - The imaginary part of the spectrum.
/// * `dim` - The dimension along which to take the inverse FFT.
/// * `n` - Optional output signal length. When `None`, the reconstructed signal length
///   `2 * (size - 1)` must be a power of two. When `Some(n)`, `n` must also be a power of
///   two and the output has exactly `n` samples. Non-power-of-two `n` is rejected.
///
/// # Returns
///
/// The reconstructed real-valued signal.
///
/// # Example
///
/// ```rust
/// use burn_tensor::backend::Backend;
/// use burn_tensor::Tensor;
///
/// fn example<B: Backend>() {
///     let device = B::Device::default();
///     let real = Tensor::<B, 1>::from_floats([10.0, -2.0, 2.0], &device);
///     let imag = Tensor::<B, 1>::from_floats([0.0, 2.0, 0.0], &device);
///     let signal = burn_tensor::signal::irfft(real, imag, 0, None);
/// }
/// ```
pub fn irfft<B: Backend, const D: usize>(
    spectrum_re: Tensor<B, D>,
    spectrum_im: Tensor<B, D>,
    dim: usize,
    n: Option<usize>,
) -> Tensor<B, D> {
    check!(TensorCheck::check_dim::<D>(dim));

    if let Some(n) = n {
        assert!(n >= 1, "irfft: n must be >= 1, got {n}");
        assert!(
            n.is_power_of_two(),
            "irfft: n must be a power of two, got {n}. True non-power-of-two \
             DFT support is tracked as a follow-up (Bluestein's algorithm)."
        );
    }

    let signal = B::irfft(
        spectrum_re.primitive.tensor(),
        spectrum_im.primitive.tensor(),
        dim,
        n,
    );
    Tensor::new(TensorPrimitive::Float(signal))
}
