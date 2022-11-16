mod base;

pub use base::*;

// Not needed for now, usefull for different tensor memory layout
// pub mod conversion;

pub(crate) mod autodiff;

#[cfg(feature = "ndarray")]
pub(crate) mod ndarray;
#[cfg(feature = "ndarray")]
pub type NdArrayADBackend<E> = self::autodiff::ADBackendNdArray<E>;
#[cfg(feature = "ndarray")]
pub type NdArrayBackend<E> = self::ndarray::NdArrayBackend<E>;
#[cfg(feature = "ndarray")]
pub type NdArrayDevice = self::ndarray::NdArrayDevice;

pub use autodiff::ADBackendDecorator;
