// Re-export
pub use burn_autodiff::{Autodiff, checkpoint::strategy::BalancedCheckpointing};
pub use burn_tensor::Tensor;

use super::FloatElemType;

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
pub type TestBackend = burn_ndarray::NdArray<FloatElemType>;

#[cfg(feature = "candle")]
pub type TestBackend = burn_candle::Candle<FloatElemType>;

#[cfg(feature = "tch")]
pub type TestBackend = burn_tch::LibTorch<FloatElemType>;

#[cfg(feature = "cuda")]
pub type TestBackend = burn_cuda::Cuda<FloatElemType, super::IntElemType>;

#[cfg(feature = "rocm")]
pub type TestBackend = burn_rocm::Rocm<FloatElemType, super::IntElemType>;

#[cfg(feature = "wgpu")]
pub type TestBackend = burn_wgpu::Wgpu<FloatElemType, super::IntElemType>;

#[cfg(feature = "cpu")]
pub type TestBackend = burn_cpu::Cpu<FloatElemType, super::IntElemType>;

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
