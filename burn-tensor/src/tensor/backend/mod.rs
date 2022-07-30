mod backend;

pub use backend::*;

pub mod autodiff;
pub mod conversion;

#[cfg(feature = "tch")]
use ::tch::Device;
#[cfg(feature = "tch")]
pub mod tch;
#[cfg(feature = "tch")]
pub type TchDevice = Device;

#[cfg(feature = "ndarray")]
pub mod ndarray;
