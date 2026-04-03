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
/// * The dimension in which the fft is done must be a power of two
///
/// # Returns
///
/// A tuple containing:
/// 1. The real part of the spectrum.
/// 2. The imaginary part of the spectrum.
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
///     let (real, imag) = burn_tensor::signal::rfft(signal, 0);
/// }
/// ```
pub fn rfft<B: Backend, const D: usize>(
    signal: Tensor<B, D>,
    dim: usize,
) -> (Tensor<B, D>, Tensor<B, D>) {
    check!(TensorCheck::check_dim::<D>(dim));
    check!(TensorCheck::check_is_power_of_two::<D>(
        &signal.shape(),
        dim
    ));
    let (spectrum_re, spectrum_im) = B::rfft(signal.primitive.tensor(), dim);
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
/// * The reconstructed signal length (2 * (size - 1)) must be a power of two.
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
///     let signal = burn_tensor::signal::irfft(real, imag, 0);
/// }
/// ```
pub fn irfft<B: Backend, const D: usize>(
    spectrum_re: Tensor<B, D>,
    spectrum_im: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    check!(TensorCheck::check_dim::<D>(dim));

    let signal = B::irfft(
        spectrum_re.primitive.tensor(),
        spectrum_im.primitive.tensor(),
        dim,
    );
    Tensor::new(TensorPrimitive::Float(signal))
}
