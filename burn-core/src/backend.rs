#[cfg(feature = "__ndarray")]
/// Ndarray module.
pub use burn_ndarray as ndarray;

#[cfg(feature = "__ndarray")]
/// An NdArrayBackend with a default type of f32.
pub type NdArrayBackend<F = f32> = ndarray::NdArray<F>;

#[cfg(all(feature = "__ndarray", feature = "autodiff"))]
/// An NdArrayBackend with autodiffing enabled.
pub type NdArrayAutodiffBackend<F = f32> = crate::autodiff::Autodiff<NdArrayBackend<F>>;

#[cfg(feature = "wgpu")]
/// WGPU module.
pub use burn_wgpu as wgpu;

#[cfg(feature = "wgpu")]
/// A WGpuBackend with a default type of f32/i32, and auto graphics.
pub type Wgpu<G = wgpu::AutoGraphicsApi, F = f32, I = i32> = wgpu::Wgpu<G, F, I>;

#[cfg(all(feature = "wgpu", feature = "autodiff"))]
/// A Wgpu with autodiffing enabled.
pub type WgpuAutodiffBackend<G = wgpu::AutoGraphicsApi, F = f32, I = i32> =
    crate::autodiff::Autodiff<Wgpu<G, F, I>>;

#[cfg(feature = "candle")]
/// Candle module.
pub use burn_candle as candle;

#[cfg(feature = "candle")]
/// A CandleBackend with a default type of f32/i64.
pub type CandleBackend = candle::CandleBackend<f32, i64>;

#[cfg(all(feature = "candle", feature = "autodiff"))]
/// A CandleBackend with autodiffing enabled.
pub type CandleAutodiffBackend = crate::autodiff::Autodiff<CandleBackend>;

#[cfg(feature = "tch")]
/// Tch module.
pub use burn_tch as tch;

#[cfg(feature = "tch")]
/// A LibTorch with a default type of f32.
pub type LibTorch<F = f32> = tch::LibTorch<F>;

#[cfg(all(feature = "tch", feature = "autodiff"))]
/// A LibTorch with autodiffing enabled.
pub type TchAutodiffBackend<F = f32> = crate::autodiff::Autodiff<LibTorch<F>>;
