use super::element::TchElement;
use super::TchTensor;
use burn_tensor::backend::Backend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// The device struct when using the `tch` backend.
///
/// Note that you need to provide the device index when using Cuda.
///
/// # Example
///
/// ```no_run
/// use burn_tch::LibTorchDevice;
///
/// let device_gpu_1 = LibTorchDevice::Cuda(0); // First GPU
/// let device_gpu_2 = LibTorchDevice::Cuda(1); // Second GPU
/// let device_cpu = LibTorchDevice::Cpu; // CPU
/// let device_mps = LibTorchDevice::Mps; // Metal Performance Shaders
/// let device_vulkan = LibTorchDevice::Vulkan; // Vulkan
/// ```
pub enum LibTorchDevice {
    /// CPU device.
    Cpu,

    /// Cuda device with the given index. The index is the index of the Cuda device in the list of
    /// all Cuda devices found on the system.
    Cuda(usize),

    /// Metal Performance Shaders device.
    Mps,

    /// Vulkan device.
    Vulkan,
}

impl From<LibTorchDevice> for tch::Device {
    fn from(device: LibTorchDevice) -> Self {
        match device {
            LibTorchDevice::Cpu => tch::Device::Cpu,
            LibTorchDevice::Cuda(num) => tch::Device::Cuda(num),
            LibTorchDevice::Mps => tch::Device::Mps,
            LibTorchDevice::Vulkan => tch::Device::Vulkan,
        }
    }
}

impl From<tch::Device> for LibTorchDevice {
    fn from(device: tch::Device) -> Self {
        match device {
            tch::Device::Cpu => LibTorchDevice::Cpu,
            tch::Device::Cuda(num) => LibTorchDevice::Cuda(num),
            tch::Device::Mps => LibTorchDevice::Mps,
            tch::Device::Vulkan => LibTorchDevice::Vulkan,
        }
    }
}

impl Default for LibTorchDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Tensor backend that uses `LibTorch` with the [tch] crate for executing tensor operations.
///
/// This backend is compatible with a wide range of hardwares ranging from CPUs to GPUs, but
/// required `LibTorch` to be installed and compiled correctly. The CPU version can be downloaded
/// automatically, but more complex configurations needs manual installation.
///
/// Refer to the [tch] crate for more information.
#[derive(Clone, Copy, Default, Debug)]
pub struct LibTorch<E = f32> {
    _e: E,
}

impl<E: TchElement> Backend for LibTorch<E> {
    type Device = LibTorchDevice;
    type FullPrecisionElem = f32;
    type FullPrecisionBackend = LibTorch<f32>;

    type TensorPrimitive<const D: usize> = TchTensor<E, D>;
    type FloatElem = E;

    type IntTensorPrimitive<const D: usize> = TchTensor<i64, D>;
    type IntElem = i64;

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

    fn sync(device: &Self::Device) {
        if let LibTorchDevice::Cuda(index) = device {
            tch::Cuda::synchronize(*index as i64);
        } else if let LibTorchDevice::Mps = device {
            panic!("Can't sync MPS device")
        }
    }
}
