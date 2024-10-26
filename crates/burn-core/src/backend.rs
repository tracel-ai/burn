#[cfg(feature = "ndarray")]
pub use burn_ndarray as ndarray;

#[cfg(feature = "ndarray")]
pub use ndarray::NdArray;

#[cfg(feature = "autodiff")]
pub use burn_autodiff as autodiff;

#[cfg(feature = "autodiff")]
pub use burn_autodiff::Autodiff;

#[cfg(feature = "wgpu")]
pub use burn_wgpu as wgpu;

#[cfg(feature = "wgpu")]
pub use burn_wgpu::Wgpu;

#[cfg(feature = "cuda-jit")]
pub use burn_cuda as cuda_jit;

#[cfg(feature = "cuda-jit")]
pub use burn_cuda::Cuda as CudaJit;

#[cfg(feature = "candle")]
pub use burn_candle as candle;

#[cfg(feature = "candle")]
pub use burn_candle::Candle;

#[cfg(feature = "hip-jit")]
pub use burn_hip as hip_jit;

#[cfg(feature = "tch")]
pub use burn_tch as libtorch;

#[cfg(feature = "tch")]
pub use burn_tch::LibTorch;
