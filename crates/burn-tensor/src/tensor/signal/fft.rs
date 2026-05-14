use alloc::vec;
use burn_backend::ops::ModuleOps;
use burn_dispatch::Dispatch;

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
/// use burn_tensor::Tensor;
///
/// fn example() {
///     let device = Default::default();
///     let signal = Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
///     let (real, imag) = burn_tensor::signal::rfft(signal, 0, None);
/// }
/// ```
pub fn rfft<const D: usize>(
    signal: Tensor<D>,
    dim: usize,
    n: Option<usize>,
) -> (Tensor<D>, Tensor<D>) {
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

    let (spectrum_re, spectrum_im) = Dispatch::rfft(signal.primitive.tensor(), dim, n);
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
/// use burn_tensor::Tensor;
///
/// fn example() {
///     let device = Default::default();
///     let real = Tensor::<1>::from_floats([10.0, -2.0, 2.0], &device);
///     let imag = Tensor::<1>::from_floats([0.0, 2.0, 0.0], &device);
///     let signal = burn_tensor::signal::irfft(real, imag, 0, None);
/// }
/// ```
pub fn irfft<const D: usize>(
    spectrum_re: Tensor<D>,
    spectrum_im: Tensor<D>,
    dim: usize,
    n: Option<usize>,
) -> Tensor<D> {
    check!(TensorCheck::check_dim::<D>(dim));

    if let Some(n) = n {
        assert!(n >= 1, "irfft: n must be >= 1, got {n}");
        assert!(
            n.is_power_of_two(),
            "irfft: n must be a power of two, got {n}. True non-power-of-two \
             DFT support is tracked as a follow-up (Bluestein's algorithm)."
        );
    }

    let signal = Dispatch::irfft(
        spectrum_re.primitive.tensor(),
        spectrum_im.primitive.tensor(),
        dim,
        n,
    );
    Tensor::new(TensorPrimitive::Float(signal))
}

/// Computes the 1-dimensional discrete Fourier Transform of complex-valued input.
///
/// Internally calls [`rfft`] on the real and imaginary parts separately,
/// extends each half-spectrum to the full `N`-bin spectrum via Hermitian
/// symmetry.
///
/// Autodiff is not yet supported.
///
#[cfg_attr(
    doc,
    doc = r#"

Due to the linearity of the Fourier Transform, a complex-valued signal $x\[n\] = x_{re}\[n\] + i x_{im}\[n\]$ can be transformed by applying the FFT to its real and imaginary parts separately:

$$ \text{FFT}(x\[n\]) = \text{FFT}(x_{re}\[n\]) + i \text{FFT}(x_{im}\[n\]) $$

Since $x_{re}\[n\]$ and $x_{im}\[n\]$ are purely real, their transforms can be computed efficiently using the real FFT ([`rfft`]). The full spectrum is then reconstructed by exploiting Hermitian symmetry.
"#
)]
#[cfg_attr(not(doc), doc = r"X\[k\] = Σ x\[n\] * exp(-i*2πkn/N)")]
///
/// # Arguments
///
/// * `signal_re` - The real part of the complex input signal.
/// * `signal_im` - The imaginary part of the complex input signal. Must have the
///   same shape as `signal_re`.
/// * `dim` - The dimension along which to take the FFT.
/// * `n` - Optional FFT length. When `None`, the signal must be a power of two
///   along `dim`. When `Some(n)`, `n` must also be a power of two; the signal is
///   truncated or zero-padded to length `n`.
///
/// # Returns
///
/// A tuple `(re, im)` representing the full complex spectrum, each with `n`
/// elements along `dim`.
///
/// # Example
///
/// ```rust
/// use burn_tensor::Tensor;
///
/// fn example() {
///     let device = Default::default();
///     let re = Tensor::<1>::from_floats([1.0, 0.0, -1.0, 0.0], &device);
///     let im = Tensor::<1>::from_floats([0.0, 1.0, 0.0, -1.0], &device);
///     let (spec_re, spec_im) = burn_tensor::signal::cfft(re, im, 0, None);
/// }
/// ```
pub fn cfft<const D: usize>(
    signal_re: Tensor<D>,
    signal_im: Tensor<D>,
    dim: usize,
    n: Option<usize>,
) -> (Tensor<D>, Tensor<D>) {
    assert!(
        signal_re.shape() == signal_im.shape(),
        "cfft: signal_re and signal_im must have the same shape, \
         got {:?} and {:?}",
        signal_re.shape(),
        signal_im.shape(),
    );

    check!(TensorCheck::check_dim::<D>(dim));
    let fft_size = n.unwrap_or(signal_re.dims()[dim]);

    // rfft validates power-of-two and n constraints internally
    let (xr, xi) = rfft(signal_re, dim, n);
    let (yr, yi) = rfft(signal_im, dim, n);

    // Extend half-spectra (N/2+1 bins) to full N-bin spectra via Hermitian symmetry
    let (xr, xi) = hermitian_extend(xr, xi, dim, fft_size);
    let (yr, yi) = hermitian_extend(yr, yi, dim, fft_size);

    // FFT(z) = FFT(x) + i·FFT(y)
    //        = (Xr + i·Xi) + i·(Yr + i·Yi)
    //        = (Xr - Yi) + i·(Xi + Yr)
    (xr - yi, xi + yr)
}

/// Extend a half-spectrum from [`rfft`] (`N/2 + 1` bins) to the full `N`-bin
/// spectrum using Hermitian symmetry: `X[k] = conj(X[N-k])` for `k > N/2`.
pub(super) fn hermitian_extend<const D: usize>(
    half_re: Tensor<D>,
    half_im: Tensor<D>,
    dim: usize,
    full_len: usize,
) -> (Tensor<D>, Tensor<D>) {
    let half_len = half_re.dims()[dim]; // N/2 + 1

    // For N <= 2, the half-spectrum already covers all bins
    if full_len <= half_len {
        return (half_re, half_im);
    }

    // Mirror bins: reverse of bins 1..N/2-1 (skipping the Nyquist bin),
    // with conjugated imaginary part. This produces X[N/2+1], X[N/2+2], ..., X[N-1]
    let mirror_len = full_len - half_len; // N/2 - 1
    let mirror_re = half_re
        .clone()
        .narrow(dim, 1, mirror_len)
        .flip([dim as isize]);
    let mirror_im = half_im
        .clone()
        .narrow(dim, 1, mirror_len)
        .flip([dim as isize])
        .neg();

    // Full spectrum = [half_spectrum, conjugate_mirror]
    let full_re = Tensor::cat(vec![half_re, mirror_re], dim);
    let full_im = Tensor::cat(vec![half_im, mirror_im], dim);

    (full_re, full_im)
}
