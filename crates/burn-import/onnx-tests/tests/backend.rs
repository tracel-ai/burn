#[cfg(feature = "backend-autodiff-wgpu")]
pub type Backend = burn_autodiff::Autodiff<burn_wgpu::Wgpu>;

#[cfg(not(feature = "backend-autodiff-wgpu"))]
pub type Backend = burn_ndarray::NdArray<f32>;
