pub use super::*;

/// Burn autodiff tests.
#[path = "autodiff/mod.rs"]
mod autodiff;

mod autodiff_checkpointing {
    pub use super::*;
    use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;

    // Override type def
    pub type TestAutodiffBackend = Autodiff<TestBackend, BalancedCheckpointing>;
    pub type TestAutodiffTensor<const D: usize> = Tensor<TestAutodiffBackend, D>;

    include!("autodiff/mod.rs");
}

/// Burn backend tensor tests.
#[path = "tensor/mod.rs"]
mod tensor;

/// Burn tensor and autodiff tests for CubeCL backends with fusion enabled.
#[cfg(feature = "cube")]
mod fusion {
    use burn_tensor::Tensor;

    pub type TestBackend = burn_fusion::Fusion<super::TestBackend>;
    pub type TestTensor<const D: usize> = Tensor<TestBackend, D>;
    pub type TestTensorInt<const D: usize> = Tensor<TestBackend, D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> = Tensor<TestBackend, D, burn_tensor::Bool>;

    // Tensor tests
    include!("tensor/mod.rs");

    // Autodiff tests
    include!("autodiff/mod.rs");

    #[cfg(test)]
    mod autodiff_checkpointing {
        pub use super::*;
        use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;

        // Override type def
        pub type TestAutodiffBackend = Autodiff<TestBackend, BalancedCheckpointing>;
        pub type TestAutodiffTensor<const D: usize> = Tensor<TestAutodiffBackend, D>;

        include!("autodiff/mod.rs");
    }
}

#[cfg(feature = "quantization")]
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
