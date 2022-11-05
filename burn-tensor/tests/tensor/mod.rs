#[cfg(feature = "ndarray")]
pub type TestBackend = burn_tensor::backend::NdArrayBackend<f32>;

#[cfg(all(feature = "tch", not(any(feature = "ndarray"))))]
pub type TestBackend = burn_tensor::backend::TchBackend<f32>;

#[cfg(feature = "ndarray")]
pub type TestADBackend = burn_tensor::backend::NdArrayADBackend<f32>;

#[cfg(all(feature = "tch", not(any(feature = "ndarray"))))]
pub type TestADBackend = burn_tensor::backend::TchADBackend<f32>;

pub type TestADTensor<const D: usize> = burn_tensor::Tensor<TestADBackend, D>;

mod activation;
mod grad;
mod module;
mod ops;
mod stats;
