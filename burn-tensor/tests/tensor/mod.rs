#[cfg(feature = "ndarray")]
pub type TestBackend = burn_tensor::backend::NdArrayBackend<f32>;

#[cfg(all(feature = "tch", not(any(feature = "ndarray"))))]
pub type TestBackend = burn_tensor::backend::TchBackend<f32>;

#[cfg(feature = "ndarray")]
pub type TestADBackend = burn_tensor::backend::NdArrayADBackend<f32>;

#[cfg(all(feature = "tch", not(any(feature = "ndarray"))))]
pub type TestADBackend = burn_tensor::backend::TchADBackend<f32>;

mod activation;
mod grad;
mod ops;
