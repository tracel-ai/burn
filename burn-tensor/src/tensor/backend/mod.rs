#[cfg(feature = "tch")]
use ::tch::Device;

pub mod autodiff;
pub mod conversion;
#[cfg(feature = "ndarray")]
pub mod ndarray;
#[cfg(feature = "tch")]
pub mod tch;

#[cfg(feature = "tch")]
pub type TchDevice = Device;
