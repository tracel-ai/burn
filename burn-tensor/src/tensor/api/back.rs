pub use crate::tensor::backend::Backend;

pub(crate) trait Allo {
    type Toto: Clone;
}

pub mod ad {
    pub use crate::tensor::backend::ADBackend as Backend;
    #[cfg(feature = "tch")]
    pub type Tch<E> = crate::tensor::backend::autodiff::ADBackendTch<E>;
    #[cfg(feature = "ndarray")]
    pub type NdArray<E> = crate::tensor::backend::autodiff::ADBackendNdArray<E>;
}

#[cfg(feature = "tch")]
pub type Tch<E> = crate::tensor::backend::tch::TchBackend<E>;
#[cfg(feature = "tch")]
pub type TchDevice = crate::tensor::backend::tch::TchDevice;

#[cfg(feature = "ndarray")]
pub type NdArray<E> = crate::tensor::backend::ndarray::NdArrayBackend<E>;
#[cfg(feature = "ndarray")]
pub type NdArrayDevice = crate::tensor::backend::ndarray::NdArrayDevice;
