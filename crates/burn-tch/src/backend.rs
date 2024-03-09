use tch::Kind;
use super::element::TchElement;
use super::{DynTchTensor, TchTensor};
use burn_tensor::backend::Backend;
use burn_tensor::{DynData, DynRankData};

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

fn dyn_rank_data_to_tch_tensor<E: tch::kind::Element>(dyn_rank_data: DynRankData<E>, device: &LibTorchDevice) -> tch::Tensor {
    tch::Tensor::from_slice(&dyn_rank_data.value).to((*device).into()).reshape(dyn_rank_data.shape.into_iter().map(|dim| dim as i64).collect::<Vec<_>>().as_slice())
}

fn tch_tensor_to_dyn_rank_data<E: tch::kind::Element + Copy>(tensor: tch::Tensor) -> DynRankData<E> {
    DynRankData::new(tensor.reshape(&[tensor.numel() as i64]).try_into().unwrap(), tensor.size().into_iter().map(|dim| dim as usize).collect())
}

/// Tensor backend that uses `LibTorch` with the [tch] crate for executing tensor operations.
///
/// This backend is compatible with a wide range of hardwares ranging from CPUs to GPUs, but
/// requires `LibTorch` to be installed correctly. The CPU version can be downloaded
/// automatically and the CUDA version as well by setting the `TORCH_CUDA_VERSION` environment
/// variable. For more complex configurations, check out the manual installation for
/// [burn-tch](https://github.com/tracel-ai/burn/tree/main/burn-tch).
///
/// Refer to the [tch] crate for more information.
#[derive(Clone, Copy, Default, Debug)]
pub struct LibTorch<E = f32> {
    _e: E,
}

impl<E: TchElement> Backend for LibTorch<E> {
    type Device = LibTorchDevice;
    type FullPrecisionBackend = LibTorch<f32>;
    type FullPrecisionElem = f32;

    type FloatTensorPrimitive<const D: usize> = TchTensor<E, D>;
    type FloatElem = E;

    type IntTensorPrimitive<const D: usize> = TchTensor<i64, D>;
    type IntElem = i64;

    type BoolTensorPrimitive<const D: usize> = TchTensor<bool, D>;

    type DynTensorPrimitive = DynTchTensor;

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        "tch".to_string()
    }

    fn seed(seed: u64) {
        tch::manual_seed(seed as i64);
    }

    fn sync(device: &Self::Device) {
        if let LibTorchDevice::Cuda(index) = device {
            tch::Cuda::synchronize(*index as i64);
        } else if let LibTorchDevice::Mps = device {
            panic!("Can't sync MPS device")
        }
    }

    fn dyn_from_data(data: DynData<Self::FullPrecisionElem, Self::IntElem>, device: &Self::Device) -> Self::DynTensorPrimitive {
        match data {
            DynData::Float(float_data) => DynTchTensor::new(dyn_rank_data_to_tch_tensor(float_data, device)),
            DynData::Int(int_data) => DynTchTensor::new(dyn_rank_data_to_tch_tensor(int_data, device)),
            DynData::Bool(bool_data) => DynTchTensor::new(dyn_rank_data_to_tch_tensor(bool_data, device)),
        }
    }

    fn dyn_into_data(dyn_tensor: Self::DynTensorPrimitive) -> DynData<Self::FullPrecisionElem, Self::IntElem> {
        match dyn_tensor.tensor.kind() {
            Kind::Float => DynData::Float(tch_tensor_to_dyn_rank_data(dyn_tensor.tensor)),
            Kind::Int64 => DynData::Int(tch_tensor_to_dyn_rank_data(dyn_tensor.tensor)),
            Kind::Bool => DynData::Bool(tch_tensor_to_dyn_rank_data(dyn_tensor.tensor)),
            _ => unreachable!()
        }
    }
}
