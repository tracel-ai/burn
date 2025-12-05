extern crate alloc;

#[cfg(test)]
mod backend {
    // Re-export
    pub use burn_autodiff::{Autodiff, checkpoint::strategy::BalancedCheckpointing};
    pub use burn_tensor::Tensor;

    // Default
    #[cfg(all(
        feature = "default",
        not(feature = "candle"),
        not(feature = "tch"),
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "wgpu"),
        not(feature = "cpu"),
        not(feature = "router")
    ))]
    pub type TestBackend = burn_ndarray::NdArray;

    #[cfg(feature = "candle")]
    pub type TestBackend = burn_candle::Candle;

    #[cfg(feature = "tch")]
    pub type TestBackend = burn_tch::LibTorch;

    #[cfg(feature = "cuda")]
    pub type TestBackend = burn_cuda::Cuda;

    #[cfg(feature = "rocm")]
    pub type TestBackend = burn_rocm::Rocm;

    #[cfg(feature = "wgpu")]
    pub type TestBackend = burn_wgpu::Wgpu;

    #[cfg(feature = "cpu")]
    pub type TestBackend = burn_cpu::Cpu;

    #[cfg(feature = "router")]
    pub type TestBackend = burn_router::BackendRouter<
        burn_router::DirectByteChannel<(burn_ndarray::NdArray, burn_wgpu::Wgpu)>,
    >;

    pub type TestTensor<const D: usize> = Tensor<TestBackend, D>;
    pub type TestTensorInt<const D: usize> = Tensor<TestBackend, D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> = Tensor<TestBackend, D, burn_tensor::Bool>;

    pub type FloatElem = burn_tensor::ops::FloatElem<TestBackend>;
    pub type IntElem = burn_tensor::ops::IntElem<TestBackend>;

    pub type TestAutodiffBackend = Autodiff<TestBackend>;
    pub type TestAutodiffTensor<const D: usize> = Tensor<TestAutodiffBackend, D>;
}

#[cfg(test)]
pub use backend::*;

/// Burn backend tensor tests.
#[cfg(test)]
mod tensor;

/// Burn autodiff tests.
#[cfg(test)]
mod autodiff;

#[cfg(test)]
mod autodiff_checkpointing {
    use super::*;

    // Override type def
    pub type TestAutodiffBackend = Autodiff<TestBackend, BalancedCheckpointing>;
    pub type TestAutodiffTensor<const D: usize> = Tensor<TestAutodiffBackend, D>;

    include!("autodiff/mod.rs");
}

/// Burn tensor and autodiff tests for CubeCL backends with fusion enabled.
#[cfg(all(test, feature = "fusion"))]
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
        use super::*;

        // Override type def
        pub type TestAutodiffBackend = Autodiff<TestBackend, BalancedCheckpointing>;
        pub type TestAutodiffTensor<const D: usize> = Tensor<TestAutodiffBackend, D>;

        include!("autodiff/mod.rs");
    }
}

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
