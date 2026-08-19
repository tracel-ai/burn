use burn_core as burn;
use burn_core::backend::{Backend, backend_extension, tensor::FloatTensor};

/// Signal processing operations supplied by a backend extension.
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(any(
        feature = "wgpu",
        feature = "webgpu",
        feature = "vulkan",
        feature = "metal",
        feature = "cuda",
        feature = "rocm",
        feature = "cpu"
    )),
    NdArray: cfg(feature = "ndarray"),
    LibTorch: cfg(feature = "tch"),
    Remote: cfg(feature = "remote"),
    Capture: cfg(feature = "capture"),
    Autodiff: cfg(feature = "autodiff"),
)]
pub trait SignalOps: Backend {
    /// Real FFT along `dim`, truncating or padding to `n` when supplied.
    /// Returns the real and imaginary components of the one-sided spectrum.
    #[allow(unused_variables)]
    fn rfft(
        signal: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);

    /// Inverse real FFT along `dim`, with optional output length `n`.
    #[allow(unused_variables)]
    fn irfft(
        real: FloatTensor<Self>,
        imag: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Self>;
}
