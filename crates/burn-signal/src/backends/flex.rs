use crate::SignalOps;
use burn_core::backend::tensor::FloatTensor;
use burn_core::{backend::Flex, tensor::DType};

impl SignalOps for Flex {
    fn rfft(
        signal: FloatTensor<Flex>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Flex>, FloatTensor<Flex>) {
        match signal.dtype() {
            DType::F32 => burn_flex::ops::fft::rfft_f32(signal, dim, n),
            DType::F64 => burn_flex::ops::fft::rfft_f64(signal, dim, n),
            DType::F16 => burn_flex::ops::fft::rfft_f16(signal, dim, n),
            DType::BF16 => burn_flex::ops::fft::rfft_bf16(signal, dim, n),
            dtype => panic!("rfft: unsupported dtype {:?}", dtype),
        }
    }

    fn irfft(
        spectrum_re: FloatTensor<Flex>,
        spectrum_im: FloatTensor<Flex>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Flex> {
        match spectrum_re.dtype() {
            DType::F32 => burn_flex::ops::fft::irfft_f32(spectrum_re, spectrum_im, dim, n),
            DType::F64 => burn_flex::ops::fft::irfft_f64(spectrum_re, spectrum_im, dim, n),
            DType::F16 => burn_flex::ops::fft::irfft_f16(spectrum_re, spectrum_im, dim, n),
            DType::BF16 => burn_flex::ops::fft::irfft_bf16(spectrum_re, spectrum_im, dim, n),
            dtype => panic!("irfft: unsupported dtype {:?}", dtype),
        }
    }
}
