#[cfg(feature = "tch")]
use ::tch::Device;

#[cfg(feature = "arrayfire")]
pub mod arrayfire;
pub mod autodiff;
pub mod conversion;
pub mod ndarray;
#[cfg(feature = "tch")]
pub mod tch;

#[cfg(feature = "tch")]
pub type TchDevice = Device;
pub enum Backend {
    #[cfg(feature = "tch")]
    Tch(TchDevice),
}

impl Default for Backend {
    fn default() -> Self {
        Backend::Tch(TchDevice::Cpu)
    }
}
