#[cfg(feature = "ndarray")]
pub use burn_ndarray as ndarray;

#[cfg(feature = "ndarray")]
pub use ndarray::NdArray;

#[cfg(feature = "autodiff")]
pub use burn_autodiff as autodiff;

#[cfg(feature = "remote")]
pub use burn_remote as remote;
#[cfg(feature = "remote")]
pub use burn_remote::RemoteBackend;

#[cfg(feature = "autodiff")]
pub use burn_autodiff::Autodiff;

#[cfg(feature = "wgpu")]
pub use burn_wgpu as wgpu;

#[cfg(feature = "wgpu")]
pub use burn_wgpu::Wgpu;

#[cfg(feature = "webgpu")]
pub use burn_wgpu::WebGpu;

#[cfg(feature = "vulkan")]
pub use burn_wgpu::Vulkan;

#[cfg(feature = "cuda")]
pub use burn_cuda as cuda;

#[cfg(feature = "cuda")]
pub use burn_cuda::Cuda;

#[cfg(feature = "candle")]
pub use burn_candle as candle;

#[cfg(feature = "candle")]
pub use burn_candle::Candle;

#[cfg(feature = "hip")]
pub use burn_hip as hip;

#[cfg(feature = "hip")]
pub use burn_hip::Hip;

#[cfg(feature = "tch")]
pub use burn_tch as libtorch;

#[cfg(feature = "tch")]
pub use burn_tch::LibTorch;

#[cfg(feature = "router")]
pub use burn_router::Router;
