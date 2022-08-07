#[cfg(feature = "ndarray")]
pub type TestBackend = burn_tensor::back::NdArray<f32>;
#[cfg(not(feature = "ndarray"))]
#[cfg(feature = "tch")]
pub type TestBackend = burn_tensor::back::Tch<f32>;

mod ops;
