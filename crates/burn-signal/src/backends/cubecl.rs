use crate::SignalOps;
use burn_core::backend::tensor::FloatTensor;
use burn_cubecl::{CubeBackend, kernel};

impl SignalOps for CubeBackend {
    fn rfft(
        signal: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        kernel::fft::rfft(signal, dim, n)
    }

    fn irfft(
        spectrum_re: FloatTensor<Self>,
        spectrum_im: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Self> {
        kernel::fft::irfft(spectrum_re, spectrum_im, dim, n)
    }
}
