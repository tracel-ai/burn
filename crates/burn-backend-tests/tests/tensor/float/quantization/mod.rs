pub use super::*; // re-export test types

mod calibration;
mod data;
mod ops;
mod scheme;

/// Quantized tensor utilities
pub mod qtensor {
    use core::marker::PhantomData;

    use burn_tensor::quantization::QuantLevel;

    use burn_tensor::{
        Tensor, TensorData,
        backend::Backend,
        quantization::{QTensorPrimitive, QuantValue},
    };

    pub struct QTensor<B: Backend, const D: usize> {
        b: PhantomData<B>,
    }

    impl<B: Backend, const D: usize> QTensor<B, D> {
        /// Creates a quantized int8 tensor from the floating point data using the default quantization scheme
        /// (i.e., per-tensor symmetric quantization).
        pub fn int8<F: Into<TensorData>>(floats: F) -> Tensor<B, D> {
            Self::int8_symmetric(floats)
        }

        /// Creates a quantized int8 tensor from the floating point data using blocks of size 16
        pub fn int8_block<F: Into<TensorData>>(floats: F) -> Tensor<B, D> {
            Tensor::from_floats(floats, &Default::default()).quantize_dynamic(
                &<B::QuantizedTensorPrimitive as QTensorPrimitive>::default_scheme()
                    .with_value(QuantValue::Q8S)
                    .with_level(QuantLevel::block([16])),
            )
        }

        /// Creates a quantized int8 tensor from the floating point data using per-tensor symmetric quantization.
        pub fn int8_symmetric<F: Into<TensorData>>(floats: F) -> Tensor<B, D> {
            Tensor::from_floats(floats, &Default::default()).quantize_dynamic(
                &<B::QuantizedTensorPrimitive as QTensorPrimitive>::default_scheme()
                    .with_value(QuantValue::Q8S),
            )
        }
    }
}
