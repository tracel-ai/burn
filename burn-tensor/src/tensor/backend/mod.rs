mod backend;

pub use backend::*;

// Not needed for now, usefull for different tensor memory layout
// pub mod conversion;

pub(crate) mod autodiff;

#[cfg(feature = "tch")]
pub(crate) mod tch;
#[cfg(feature = "tch")]
pub type TchADBackend<E> = self::autodiff::ADBackendTch<E>;
#[cfg(feature = "tch")]
pub type TchBackend<E> = self::tch::TchBackend<E>;
#[cfg(feature = "tch")]
pub type TchDevice = self::tch::TchDevice;

#[cfg(feature = "ndarray")]
pub(crate) mod ndarray;
#[cfg(feature = "ndarray")]
pub type NdArrayADBackend<E> = self::autodiff::ADBackendNdArray<E>;
#[cfg(feature = "ndarray")]
pub type NdArrayBackend<E> = self::ndarray::NdArrayBackend<E>;
#[cfg(feature = "ndarray")]
pub type NdArrayDevice = self::ndarray::NdArrayDevice;
