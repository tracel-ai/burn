use super::element::TchElement;
use super::TchTensor;
use burn_tensor::backend::Backend;

#[derive(Clone, Copy, Debug)]
/// The device struct when using the `tch` backend.
///
/// Note that you need to provide the device index when using Cuda.
///
/// # Example
///
/// ```rust
/// use burn_tch::TchDevice;
///
/// let device_gpu_1 = TchDevice::Cuda(0); // First GPU
/// let device_gpu_2 = TchDevice::Cuda(1); // Second GPU
/// let device_cpu = TchDevice::Cpu; // CPU
/// let device_mps = TchDevice::Mps; // Metal Performance Shaders
/// ```
pub enum TchDevice {
    Cpu,
    Cuda(usize),
    Mps,
}

impl From<TchDevice> for tch::Device {
    fn from(device: TchDevice) -> Self {
        match device {
            TchDevice::Cpu => tch::Device::Cpu,
            TchDevice::Cuda(num) => tch::Device::Cuda(num),
            TchDevice::Mps => tch::Device::Mps,
        }
    }
}

impl From<tch::Device> for TchDevice {
    fn from(device: tch::Device) -> Self {
        match device {
            tch::Device::Cpu => TchDevice::Cpu,
            tch::Device::Cuda(num) => TchDevice::Cuda(num),
            tch::Device::Mps => TchDevice::Mps,
        }
    }
}

impl Default for TchDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct TchBackend<E> {
    _e: E,
}

impl<E: TchElement> Backend for TchBackend<E> {
    type Device = TchDevice;
    type Elem = E;
    type FullPrecisionElem = f32;
    type FullPrecisionBackend = TchBackend<f32>;
    type IntegerBackend = TchBackend<i64>;
    type TensorPrimitive<const D: usize> = TchTensor<E, D>;
    type BoolTensorPrimitive<const D: usize> = TchTensor<bool, D>;

    fn seed(seed: u64) {
        tch::manual_seed(seed as i64);
    }

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        "tch".to_string()
    }
}
