/// Ndarray module.
#[cfg(feature = "ndarray")]
pub use burn_ndarray as ndarray;
#[cfg(feature = "ndarray")]
/// An NdArrayBackend with a default type of f32.
pub type NdArrayBackend<F = f32> = ndarray::NdArrayBackend<F>;
/// An NdArrayBackend with autodiffing enabled.
#[cfg(all(feature = "ndarray", feature = "autodiff"))]
pub type NdArrayAutodiffBackend<F = f32> = crate::autodiff::ADBackendDecorator<NdArrayBackend<F>>;

/// WGPU module.
#[cfg(feature = "wgpu")]
pub use burn_wgpu as wgpu;
#[cfg(feature = "wgpu")]
/// A WGpuBackend with a default type of f32/i32, and auto graphics.
pub type WgpuBackend<G = wgpu::AutoGraphicsApi, F = f32, I = i32> = wgpu::WgpuBackend<G, F, I>;
/// A WgpuBackend with autodiffing enabled.
#[cfg(all(feature = "wgpu", feature = "autodiff"))]
pub type WgpuAutodiffBackend<G = wgpu::AutoGraphicsApi, F = f32, I = i32> =
    crate::autodiff::ADBackendDecorator<WgpuBackend<G, F, I>>;

/// Tch module.
#[cfg(feature = "tch")]
pub use burn_tch as tch;
/// A TchBackend with a default type of f32.
#[cfg(feature = "tch")]
pub type TchBackend<F = f32> = tch::TchBackend<F>;
/// A TchBackend with autodiffing enabled.
#[cfg(all(feature = "tch", feature = "autodiff"))]
pub type TchAutodiffBackend<F = f32> = crate::autodiff::ADBackendDecorator<TchBackend<F>>;

/// Candle module.
#[cfg(feature = "candle")]
pub use burn_candle as candle;
/// A CandleBackend with a default type of f32/i32.
#[cfg(feature = "candle")]
pub type CandleBackend<F = f32, I = i32> = candle::CandleBackend<F, I>;
/// A CandleBackend with autodiffing enabled.
#[cfg(all(feature = "candle", feature = "autodiff"))]
pub type CandleAutodiffBackend<F = f32, I = i32> =
    crate::autodiff::ADBackendDecorator<CandleBackend<F, I>>;
