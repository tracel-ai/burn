use crate::SignalOps;
use burn_core::backend::NdArray;
use burn_core::backend::tensor::FloatTensor;

impl SignalOps for NdArray {
    fn rfft(
        _signal: FloatTensor<Self>,
        _dim: usize,
        _n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        todo!("rfft is not supported for ndarray")
    }

    fn irfft(
        _spectrum_re: FloatTensor<Self>,
        _spectrum_im: FloatTensor<Self>,
        _dim: usize,
        _n: Option<usize>,
    ) -> FloatTensor<Self> {
        todo!("irfft is not supported for ndarray")
    }
}
