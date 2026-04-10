#[cfg(feature = "flex")]
pub use burn_flex as flex;

#[cfg(feature = "flex")]
pub use flex::Flex;

#[cfg(feature = "ndarray")]
pub use burn_ndarray as ndarray;

// #[cfg(feature = "remote")]
// pub use burn_remote as remote;
// #[cfg(feature = "remote")]
// pub use burn_remote::RemoteBackend;

#[cfg(feature = "wgpu")]
pub use burn_wgpu as wgpu;

#[cfg(feature = "cuda")]
pub use burn_cuda as cuda;

#[cfg(feature = "candle")]
pub use burn_candle as candle;

#[cfg(feature = "rocm")]
pub use burn_rocm as rocm;

#[cfg(feature = "tch")]
pub use burn_tch as libtorch;

// #[cfg(feature = "router")]
// pub use burn_router::Router;
// #[cfg(feature = "router")]
// pub use burn_router as router;

#[cfg(feature = "ir")]
pub use burn_ir as ir;

#[cfg(feature = "cpu")]
pub use burn_cpu as cpu;
