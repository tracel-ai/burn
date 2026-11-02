pub use super::*; // re-export test types

mod accuracy;
mod calibration;
mod data;
mod ops;
mod scheme;

/// Quantized tensor utilities
pub mod qtensor {
    use super::TestTensor;

    use burn_tensor::{
        DType, TensorData,
        quantization::{QuantValue, ScaleDtype},
    };

    pub struct QTensor<const D: usize>;

    /// `expected` as an operation on a quantized tensor returns it: quantized again under the
    /// scheme of `output`, since an operation's output keeps its quantized input's format.
    ///
    /// # Panics
    ///
    /// If `output` is not quantized.
    pub fn requantized<const D: usize>(
        expected: impl Into<TensorData>,
        output: &TestTensor<D>,
    ) -> TensorData {
        let DType::QFloat(scheme) = output.dtype() else {
            panic!(
                "an operation on a quantized tensor stays quantized, got {:?}",
                output.dtype()
            )
        };
        TestTensor::<D>::from_data(expected.into(), &output.device())
            .quantize_dynamic(&scheme)
            .dequantize()
            .into_data()
    }

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
                    .settings()
                    .quantization
                    .scheme
                    .with_value(QuantValue::Q8S)
                    .per_block([16], ScaleDtype::F32),
            )
        }

        /// Creates a quantized int8 tensor from the floating point data using per-tensor symmetric quantization.
        pub fn int8_symmetric<F: Into<TensorData>>(floats: F) -> TestTensor<D> {
            let device = Default::default();
            TestTensor::from_data(floats, &device).quantize_dynamic(
                &device
                    .settings()
                    .quantization
                    .scheme
                    .with_value(QuantValue::Q8S),
            )
        }
    }
}
