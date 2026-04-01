use burn_backend::Backend;

use crate::Tensor;
use crate::TensorPrimitive;

/// Computes the 1-dimensional discrete Fourier Transform of real-valued input.
///
/// Since the input is real, the Hermitian symmetry is exploited, and only the
/// first non-redundant values are returned ($N/2 + 1$).
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
    // options: impl Into<TensorCreationOptions<B>>,
) -> (Tensor<B, D>, Tensor<B, D>) {
    let (spectrum_re, spectrum_im) = B::rfft(signal.primitive.tensor(), dim);
    (
        Tensor::new(TensorPrimitive::Float(spectrum_re)),
        Tensor::new(TensorPrimitive::Float(spectrum_im)),
    )
}
