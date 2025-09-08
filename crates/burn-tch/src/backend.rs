use super::TchTensor;
use super::element::TchElement;
use burn_tensor::backend::{Backend, DeviceId, DeviceOps};
use burn_tensor::ops::IntTensorOps;
use burn_tensor::{Int, Tensor};

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
    #[allow(
        unreachable_code,
        reason = "CUDA branch always panics if the library is missing"
    )]
    fn from(device: LibTorchDevice) -> Self {
        match device {
            LibTorchDevice::Cpu => tch::Device::Cpu,
            LibTorchDevice::Cuda(_num) => {
                include!(concat!(env!("OUT_DIR"), "/tch_gpu_check.rs"));
                tch::Device::Cuda(_num)
            }
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

impl DeviceOps for LibTorchDevice {
    fn id(&self) -> burn_tensor::backend::DeviceId {
        match self {
            LibTorchDevice::Cpu => DeviceId::new(0, 0),
            LibTorchDevice::Cuda(index) => DeviceId::new(1, *index as u32),
            LibTorchDevice::Mps => DeviceId::new(2, 0),
            LibTorchDevice::Vulkan => DeviceId::new(3, 0),
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
/// requires `LibTorch` to be installed correctly. The CPU version can be downloaded
/// automatically and the CUDA version as well by setting the `TORCH_CUDA_VERSION` environment
/// variable. For more complex configurations, check out the manual installation for
/// [burn-tch](https://github.com/tracel-ai/burn/tree/main/crates/burn-tch).
///
/// Refer to the [tch] crate for more information.
#[derive(Clone, Copy, Default, Debug)]
pub struct LibTorch<E = f32> {
    _e: E,
}

impl<E: TchElement> Backend for LibTorch<E> {
    type Device = LibTorchDevice;

    type FloatTensorPrimitive = TchTensor;
    type FloatElem = E;

    type IntTensorPrimitive = TchTensor;
    type IntElem = i64;

    type BoolTensorPrimitive = TchTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = TchTensor;

    fn seed(_device: &Self::Device, seed: u64) {
        tch::manual_seed(seed as i64);
    }

    fn ad_enabled() -> bool {
        false
    }

    fn name(device: &Self::Device) -> String {
        match device {
            LibTorchDevice::Cpu => "libtorch<cpu>",
            LibTorchDevice::Cuda(_) => "libtorch<cuda>",
            LibTorchDevice::Mps => "libtorch<metal>",
            LibTorchDevice::Vulkan => "libtorch<vulkan>",
        }
        .to_string()
    }

    fn sync(device: &Self::Device) {
        match device {
            LibTorchDevice::Cpu => (),
            LibTorchDevice::Cuda(index) => {
                tch::Cuda::synchronize(*index as i64);
            }
            _ => {
                // When there is no explicit way to synchronize, we write and read one value to sync
                Tensor::<Self, 1, Int>::from_primitive(<Self as IntTensorOps<Self>>::int_zeros(
                    [1].into(),
                    device,
                    E::dtype().into(),
                ))
                .into_data();
            }
        }
    }
}
