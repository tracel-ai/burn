#[cfg(feature = "test-wgpu")]
pub type TestBackend = burn::backend::Wgpu;

#[cfg(all(
    feature = "test-ndarray",
    not(feature = "test-wgpu"),
    not(feature = "test-tch")
))]
pub type TestBackend = burn::backend::NdArray<f32>;

#[cfg(feature = "test-tch")]
pub type TestBackend = burn::backend::LibTorch<f32>;
