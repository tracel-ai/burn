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

#[cfg(feature = "metal")]
pub use burn_wgpu::Metal;

#[cfg(feature = "cuda")]
pub use burn_cuda as cuda;

#[cfg(feature = "cuda")]
pub use burn_cuda::Cuda;

#[cfg(feature = "candle")]
pub use burn_candle as candle;

#[cfg(feature = "candle")]
pub use burn_candle::Candle;

#[cfg(feature = "rocm")]
pub use burn_rocm as rocm;

#[cfg(feature = "rocm")]
pub use burn_rocm::Rocm;

#[cfg(feature = "tch")]
pub use burn_tch as libtorch;

#[cfg(feature = "tch")]
pub use burn_tch::LibTorch;

#[cfg(feature = "router")]
pub use burn_router::Router;

#[cfg(feature = "router")]
pub use burn_router as router;

#[cfg(feature = "ir")]
pub use burn_ir as ir;

#[cfg(feature = "collective")]
pub use burn_collective as collective;
#[cfg(feature = "cpu")]
pub use burn_cpu as cpu;

#[cfg(feature = "cpu")]
pub use burn_cpu::Cpu;
