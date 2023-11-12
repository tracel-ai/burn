#[cfg(feature = "__ndarray")]
pub use burn_ndarray as ndarray;

#[cfg(feature = "__ndarray")]
pub use ndarray::NdArray;

#[cfg(feature = "autodiff")]
pub use burn_autodiff as autodiff;

#[cfg(feature = "autodiff")]
pub use burn_autodiff::Autodiff;

#[cfg(feature = "fusion")]
pub use burn_fusion::Fusion;

#[cfg(feature = "wgpu")]
pub use burn_wgpu as wgpu;

#[cfg(feature = "wgpu")]
pub use burn_wgpu::Wgpu;

#[cfg(feature = "candle")]
pub use burn_candle as candle;

#[cfg(feature = "candle")]
pub use burn_candle::Candle;

#[cfg(feature = "tch")]
pub use burn_tch as libtorch;

#[cfg(feature = "tch")]
pub use burn_tch::LibTorch;
