#[cfg(feature = "backend-wgpu")]
pub type Backend = burn::backend::Wgpu;

#[cfg(all(
    feature = "backend-ndarray",
    not(feature = "backend-wgpu"),
    not(feature = "backend-tch")
))]
pub type Backend = burn::backend::NdArray<f32>;

#[cfg(feature = "backend-tch")]
pub type Backend = burn::backend::LibTorch<f32>;
