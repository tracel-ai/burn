#[cfg(feature = "ndarray")]
pub type TestBackend = burn_tensor::back::NdArray<f32>;

#[cfg(all(feature = "tch", not(any(feature = "ndarray"))))]
pub type TestBackend = burn_tensor::back::Tch<f32>;

#[cfg(feature = "ndarray")]
pub type TestADBackend = burn_tensor::back::ad::NdArray<f32>;

#[cfg(all(feature = "tch", not(any(feature = "ndarray"))))]
pub type TestADBackend = burn_tensor::back::ad::Tch<f32>;

mod activation;
mod grad;
mod ops;
