#[cfg(all(feature = "ndarray", not(any(feature = "tch"))))]
pub type TestBackend = burn_tensor::back::NdArray<f32>;

#[cfg(all(feature = "tch", not(any(feature = "ndarray"))))]
pub type TestBackend = burn_tensor::back::Tch<f32>;

mod ops;
