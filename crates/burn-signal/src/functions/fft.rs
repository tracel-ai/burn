use crate::SignalOps;
use alloc::vec;
use alloc::vec::Vec;
use burn_core::{
    backend::Dispatch,
    tensor::{AsIndex, DType, Device, Tensor, TensorData, ops::PadMode},
};

/// Computes the 1-dimensional discrete Fourier Transform of real-valued input.
///
/// Since the input is real, the Hermitian symmetry is exploited, and only the
/// first non-redundant values are returned ($N/2 + 1$).
/// Autodiff is supported when the `autodiff` feature is enabled.
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
///   Negative dimensions are supported and count from the end.
/// * `n` - Optional FFT length. When `None`, the signal length along `dim` is used.
///   When `Some(n)`, the signal is truncated or zero-padded to length `n`.
///   Arbitrary `n` is supported: power-of-two sizes use the radix-2 backend, and
///   other sizes fall back to Bluestein's chirp-z algorithm.
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
/// ```rust,no_run
/// use burn_core::tensor::Tensor;
///
/// let device = Default::default();
/// let signal = Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
/// let (real, imag) = burn_signal::rfft(signal, 0, None);
/// ```
pub fn rfft<const D: usize>(
    signal: Tensor<D>,
    dim: impl AsIndex,
    n: Option<usize>,
) -> (Tensor<D>, Tensor<D>) {
    let dim = dim
        .try_dim_index(D)
        .unwrap_or_else(|error| panic!("RFFT: {error}"));
    let fft_size = n.unwrap_or(signal.dims()[dim]);
    assert!(fft_size >= 1, "rfft: n must be >= 1, got {fft_size}");

    if !fft_size.is_power_of_two() {
        let zeros = Tensor::zeros_like(&signal);
        let (re, im) = bluestein_dft(signal, zeros, dim, fft_size);
        let half = fft_size / 2 + 1;
        return (re.narrow(dim, 0, half), im.narrow(dim, 0, half));
    }

    let (re, im) = <Dispatch as SignalOps>::rfft(signal.dequantize().into_dispatch(), dim, n);
    (Tensor::from_dispatch(re), Tensor::from_dispatch(im))
}

/// Computes the 1-dimensional inverse discrete Fourier Transform for real-valued signals.
///
/// This function reconstructs the real-valued time-domain signal from the
/// first non-redundant values ($N/2 + 1$) of the frequency-domain spectrum.
/// Autodiff is supported when the `autodiff` feature is enabled.
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
///   Negative dimensions are supported and count from the end.
/// * `n` - Optional output signal length. When `None`, the reconstructed signal length
///   `2 * (size - 1)` is used. When `Some(n)`, the output has exactly `n` samples.
///   Arbitrary `n` is supported: power-of-two sizes use the radix-2 backend, and
///   other sizes fall back to Bluestein's chirp-z algorithm.
///
/// # Returns
///
/// The reconstructed real-valued signal.
///
/// # Example
///
/// ```rust,no_run
/// use burn_core::tensor::Tensor;
///
/// let device = Default::default();
/// let real = Tensor::<1>::from_floats([10.0, -2.0, 2.0], &device);
/// let imag = Tensor::<1>::from_floats([0.0, 2.0, 0.0], &device);
/// let signal = burn_signal::irfft(real, imag, 0, None);
/// ```
pub fn irfft<const D: usize>(
    spectrum_re: Tensor<D>,
    spectrum_im: Tensor<D>,
    dim: impl AsIndex,
    n: Option<usize>,
) -> Tensor<D> {
    let dim = dim
        .try_dim_index(D)
        .unwrap_or_else(|error| panic!("IRFFT: {error}"));

    assert!(
        spectrum_re.shape() == spectrum_im.shape(),
        "irfft: spectrum_re and spectrum_im must have the same shape, \
         got {:?} and {:?}",
        spectrum_re.shape(),
        spectrum_im.shape(),
    );

    if let Some(n) = n {
        assert!(n >= 1, "irfft: n must be >= 1, got {n}");
    }
    let bins = spectrum_re.dims()[dim];
    assert!(bins >= 1, "irfft: spectrum dimension cannot be empty");
    let out_len = n.unwrap_or((bins - 1) * 2);
    assert!(
        out_len >= 1,
        "irfft: reconstructed signal length must be >= 1, got {out_len}"
    );

    if !out_len.is_power_of_two() {
        // Rebuild the full Hermitian spectrum, then invert it through the generic
        // Bluestein path: x = conj(DFT(conj(X))) / n.
        let half = out_len / 2 + 1;
        let re = resize_dim(spectrum_re, dim, half);
        let im = resize_dim(spectrum_im, dim, half);
        let (full_re, full_im) = hermitian_extend(re, im, dim, out_len);
        let (x_re, _) = bluestein_dft(full_re, full_im.neg(), dim, out_len);
        return x_re.mul_scalar(1.0 / out_len as f64);
    }

    Tensor::from_dispatch(<Dispatch as SignalOps>::irfft(
        spectrum_re.dequantize().into_dispatch(),
        spectrum_im.dequantize().into_dispatch(),
        dim,
        n,
    ))
}

// ============================================================================
// Bluestein's chirp-z algorithm (arbitrary-length DFT)
// ============================================================================

/// Truncate or zero-pad `tensor` so that `dim` has exactly `len` elements.
fn resize_dim<const D: usize>(tensor: Tensor<D>, dim: usize, len: usize) -> Tensor<D> {
    let current = tensor.dims()[dim];
    if current == len {
        tensor
    } else if current > len {
        tensor.narrow(dim, 0, len)
    } else {
        let mut padding = vec![(0usize, 0usize); D];
        padding[dim] = (0, len - current);
        tensor.pad(&padding[..], PadMode::Constant(0.0))
    }
}

/// Evaluates `exp(-i * pi * k^2 / n)`.
///
/// `k^2` is reduced modulo `2n` before scaling so the phase argument stays in `[0, 2*pi)`.
fn chirp_phase(k: usize, n: usize) -> (f64, f64) {
    let r = (k as u128 * k as u128 % (2 * n as u128)) as f64;
    let angle = core::f64::consts::PI * r / n as f64;
    (libm::cos(angle), -libm::sin(angle))
}

/// Builds a constant tensor of length `len` along `dim` (all other dimensions are `1`).
fn broadcast_const<const D: usize>(
    values: Vec<f64>,
    len: usize,
    dim: usize,
    device: &Device,
    dtype: DType,
) -> Tensor<D> {
    let data = TensorData::new(values, [len]).convert_dtype(dtype);
    let tensor: Tensor<1> = Tensor::from_data(data, (device, dtype));
    let mut shape = [1usize; D];
    shape[dim] = len;
    tensor.reshape(shape)
}

/// Forward complex DFT of size `n` via Bluestein's chirp-z transform.
///
/// The input is truncated or zero-padded to `n` along `dim` first. The convolution that
/// implements the transform is evaluated with `m = next_pow2(2n - 1)` point transforms,
/// so the backend's radix-2 FFT is reused for any `n`.
fn bluestein_dft<const D: usize>(
    re: Tensor<D>,
    im: Tensor<D>,
    dim: usize,
    n: usize,
) -> (Tensor<D>, Tensor<D>) {
    debug_assert!(n >= 1);

    let device = re.device();
    let dtype = re.dtype();
    let m = (2 * n - 1).next_power_of_two();

    let re = resize_dim(re, dim, n);
    let im = resize_dim(im, dim, n);

    // Chirp sequence w[k] = exp(-i*pi*k^2/n).
    let (w_re, w_im) = chirp_sequence(n, dim, &device, dtype);

    // a[k] = x[k] * w[k]
    let a_re = re.clone() * w_re.clone() - im.clone() * w_im.clone();
    let a_im = re * w_im.clone() + im * w_re.clone();

    let a_re = resize_dim(a_re, dim, m);
    let a_im = resize_dim(a_im, dim, m);

    let (b_re, b_im) = kernel_sequence(m, n, dim, &device, dtype);

    // Circular convolution of a and b through power-of-two transforms.
    let (fa_re, fa_im) = cfft(a_re, a_im, dim, Some(m));
    let (fb_re, fb_im) = cfft(b_re, b_im, dim, Some(m));

    let conv_re = fa_re.clone() * fb_re.clone() - fa_im.clone() * fb_im.clone();
    let conv_im = fa_re * fb_im + fa_im * fb_re;

    // ifft(conv) = conj(fft(conj(conv))) / m
    let (ifft_re, ifft_im) = cfft(conv_re, conv_im.neg(), dim, Some(m));
    let scale = 1.0 / m as f64;
    let y_re = ifft_re.mul_scalar(scale).narrow(dim, 0, n);
    let y_im = ifft_im.neg().mul_scalar(scale).narrow(dim, 0, n);

    // X[k] = w[k] * (a (*) b)[k]
    let x_re = y_re.clone() * w_re.clone() - y_im.clone() * w_im.clone();
    let x_im = y_re * w_im + y_im * w_re;

    (x_re, x_im)
}

/// Chirp sequence `w[k] = exp(-i*pi*k^2/n)` for `k in 0..n`.
fn chirp_sequence<const D: usize>(
    n: usize,
    dim: usize,
    device: &Device,
    dtype: DType,
) -> (Tensor<D>, Tensor<D>) {
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for k in 0..n {
        let (c, s) = chirp_phase(k, n);
        re.push(c);
        im.push(s);
    }
    (
        broadcast_const(re, n, dim, device, dtype),
        broadcast_const(im, n, dim, device, dtype),
    )
}

/// Filter kernel `b[k] = conj(w[k])`, laid out cyclically over `m` samples:
/// `b[0] = 1`, `b[k] = conj(w[k])`, and `b[m - k] = conj(w[k])` for `k in 1..n`.
fn kernel_sequence<const D: usize>(
    m: usize,
    n: usize,
    dim: usize,
    device: &Device,
    dtype: DType,
) -> (Tensor<D>, Tensor<D>) {
    let mut re = vec![0.0f64; m];
    let mut im = vec![0.0f64; m];
    for k in 0..n {
        let (c, s) = chirp_phase(k, n);
        // conj(w[k]) = cos + i*sin
        re[k] = c;
        im[k] = -s;
        if k > 0 {
            re[m - k] = c;
            im[m - k] = -s;
        }
    }
    (
        broadcast_const(re, m, dim, device, dtype),
        broadcast_const(im, m, dim, device, dtype),
    )
}

/// Computes the 1-dimensional discrete Fourier Transform of complex-valued input.
///
/// Internally calls [`rfft`] on the real and imaginary parts separately,
/// extends each half-spectrum to the full `N`-bin spectrum via Hermitian
/// symmetry.
///
/// Autodiff is supported when the `autodiff` feature is enabled.
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
///   Negative dimensions are supported and count from the end.
/// * `n` - Optional FFT length. When `None`, the signal length along `dim` is used.
///   When `Some(n)`, the signal is truncated or zero-padded to length `n`.
///   Arbitrary `n` is supported (see [`rfft`]).
///
/// # Returns
///
/// A tuple `(re, im)` representing the full complex spectrum, each with `n`
/// elements along `dim`.
///
/// # Example
///
/// ```rust,no_run
/// use burn_core::tensor::Tensor;
///
/// let device = Default::default();
/// let re = Tensor::<1>::from_floats([1.0, 0.0, -1.0, 0.0], &device);
/// let im = Tensor::<1>::from_floats([0.0, 1.0, 0.0, -1.0], &device);
/// let (spec_re, spec_im) = burn_signal::cfft(re, im, 0, None);
/// ```
pub fn cfft<const D: usize>(
    signal_re: Tensor<D>,
    signal_im: Tensor<D>,
    dim: impl AsIndex,
    n: Option<usize>,
) -> (Tensor<D>, Tensor<D>) {
    assert!(
        signal_re.shape() == signal_im.shape(),
        "cfft: signal_re and signal_im must have the same shape, \
         got {:?} and {:?}",
        signal_re.shape(),
        signal_im.shape(),
    );

    let dim = dim
        .try_dim_index(D)
        .unwrap_or_else(|error| panic!("CFFT: {error}"));
    let fft_size = n.unwrap_or(signal_re.dims()[dim]);

    // rfft handles arbitrary n (power-of-two via the backend, otherwise Bluestein)
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
