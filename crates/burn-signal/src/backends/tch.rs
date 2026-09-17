use crate::SignalOps;
use burn_core::backend::tensor::FloatTensor;
use burn_tch::{LibTorch, TchTensor};

impl SignalOps for LibTorch {
    fn rfft(
        signal: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let complex = signal
            .tensor
            .fft_rfft(n.map(|v| v as i64), dim as i64, "backward");
        let re = TchTensor::new(complex.real().contiguous());
        let im = TchTensor::new(complex.imag().contiguous());
        (re, im)
    }

    fn irfft(
        spectrum_re: FloatTensor<Self>,
        spectrum_im: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Self> {
        let complex = tch::Tensor::complex(&spectrum_re.tensor, &spectrum_im.tensor);
        TchTensor::new(complex.fft_irfft(n.map(|v| v as i64), dim as i64, "backward"))
    }
}
