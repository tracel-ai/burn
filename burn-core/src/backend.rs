/// Ndarray module.
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
#[cfg(feature = "ndarray")]
pub use burn_ndarray as ndarray;

#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
#[cfg(feature = "ndarray")]
/// An NdArrayBackend with a default type of f32.
pub type NdArrayBackend<F = f32> = ndarray::NdArrayBackend<F>;

#[cfg_attr(docsrs, doc(cfg(all(feature = "ndarray", feature = "autodiff"))))]
#[cfg(all(feature = "ndarray", feature = "autodiff"))]
/// An NdArrayBackend with autodiffing enabled.
pub type NdArrayAutodiffBackend<F = f32> = crate::autodiff::ADBackendDecorator<NdArrayBackend<F>>;

/// WGPU module.
#[cfg_attr(docsrs, doc(cfg(feature = "wgpu")))]
#[cfg(feature = "wgpu")]
pub use burn_wgpu as wgpu;
#[cfg_attr(docsrs, doc(cfg(feature = "wpgu")))]
#[cfg(feature = "wgpu")]
/// A WGpuBackend with a default type of f32/i32, and auto graphics.
pub type WgpuBackend<G = wgpu::AutoGraphicsApi, F = f32, I = i32> = wgpu::WgpuBackend<G, F, I>;
/// A WgpuBackend with autodiffing enabled.
#[cfg_attr(docsrs, doc(cfg(all(feature = "wgpu", feature = "autodiff"))))]
#[cfg(all(feature = "wgpu", feature = "autodiff"))]
pub type WgpuAutodiffBackend<G = wgpu::AutoGraphicsApi, F = f32, I = i32> =
    crate::autodiff::ADBackendDecorator<WgpuBackend<G, F, I>>;

/// Tch module.
#[cfg_attr(docsrs, doc(cfg(feature = "tch")))]
#[cfg(feature = "tch")]
pub use burn_tch as tch;
/// A TchBackend with a default type of f32.
#[cfg_attr(docsrs, doc(cfg(feature = "tch")))]
#[cfg(feature = "tch")]
pub type TchBackend<F = f32> = tch::TchBackend<F>;
/// A TchBackend with autodiffing enabled.
#[cfg_attr(docsrs, doc(cfg(all(feature = "tch", feature = "autodiff"))))]
#[cfg(all(feature = "tch", feature = "autodiff"))]
pub type TchAutodiffBackend<F = f32> = crate::autodiff::ADBackendDecorator<TchBackend<F>>;
