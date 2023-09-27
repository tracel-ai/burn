#[cfg(feature = "__ndarray")]
/// Ndarray module.
pub use burn_ndarray as ndarray;

#[cfg(feature = "__ndarray")]
/// An NdArrayBackend with a default type of f32.
pub type NdArrayBackend<F = f32> = ndarray::NdArrayBackend<F>;

#[cfg(all(feature = "__ndarray", feature = "autodiff"))]
/// An NdArrayBackend with autodiffing enabled.
pub type NdArrayAutodiffBackend<F = f32> = crate::autodiff::ADBackendDecorator<NdArrayBackend<F>>;

#[cfg(feature = "wgpu")]
/// WGPU module.
pub use burn_wgpu as wgpu;

#[cfg(feature = "wgpu")]
/// A WGpuBackend with a default type of f32/i32, and auto graphics.
pub type WgpuBackend<G = wgpu::AutoGraphicsApi, F = f32, I = i32> = wgpu::WgpuBackend<G, F, I>;

#[cfg(all(feature = "wgpu", feature = "autodiff"))]
/// A WgpuBackend with autodiffing enabled.
pub type WgpuAutodiffBackend<G = wgpu::AutoGraphicsApi, F = f32, I = i32> =
    crate::autodiff::ADBackendDecorator<WgpuBackend<G, F, I>>;

#[cfg(feature = "tch")]
/// Tch module.
pub use burn_tch as tch;

#[cfg(feature = "tch")]
/// A TchBackend with a default type of f32.
pub type TchBackend<F = f32> = tch::TchBackend<F>;

#[cfg(all(feature = "tch", feature = "autodiff"))]
/// A TchBackend with autodiffing enabled.
pub type TchAutodiffBackend<F = f32> = crate::autodiff::ADBackendDecorator<TchBackend<F>>;
