pub use super::*; // re-export test types

mod calibration;
mod data;
mod ops;
mod scheme;

/// Quantized tensor utilities
pub mod qtensor {
    use super::TestTensor;

    use burn_tensor::quantization::QuantLevel;
    use burn_tensor::{TensorData, quantization::QuantValue};

    pub struct QTensor<const D: usize>;

    impl<const D: usize> QTensor<D> {
        /// Creates a quantized int8 tensor from the floating point data using the default quantization scheme
        /// (i.e., per-tensor symmetric quantization).
        pub fn int8<F: Into<TensorData>>(floats: F) -> TestTensor<D> {
            Self::int8_symmetric(floats)
        }

        /// Creates a quantized int8 tensor from the floating point data using blocks of size 16
        pub fn int8_block<F: Into<TensorData>>(floats: F) -> TestTensor<D> {
            let device = Default::default();
            TestTensor::from_data(floats, &device).quantize_dynamic(
                &device
                    .default_quant_scheme()
                    .with_value(QuantValue::Q8S)
                    .with_level(QuantLevel::block([16])),
            )
        }

        /// Creates a quantized int8 tensor from the floating point data using per-tensor symmetric quantization.
        pub fn int8_symmetric<F: Into<TensorData>>(floats: F) -> TestTensor<D> {
            let device = Default::default();
            TestTensor::from_data(floats, &device)
                .quantize_dynamic(&device.default_quant_scheme().with_value(QuantValue::Q8S))
        }
    }
}
