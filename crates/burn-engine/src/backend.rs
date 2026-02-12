use alloc::format;
use alloc::string::String;

use burn_backend::Backend;
use burn_backend::ExecutionError;
use burn_std::DType;

use crate::backends::*;
use crate::{Device, EngineTensor, dispatch_device};

/// The main execution engine in Burn.
///
/// [`Engine`] acts as a global backend that can manage multiple underlying
/// backends (e.g., `Cpu`, `Cuda`, `Wgpu`, `Metal`, etc.).  
/// It is responsible for:
/// - Dispatching tensor operations to the appropriate backend.
/// - Managing cross-backend tensor transfers.
///
/// Essentially, [`Engine`] is the single entry point for executing tensor operations
/// in a backend-agnostic way. It allows Burn to provide a unified, global backend
/// for users while still leveraging multiple specialized backends under the hood.
///
/// # Example
///
/// ```ignore
/// use burn::Engine;
/// use burn::Device;
///
/// // Select the device to execute operations on
/// let device = Device::Cuda(Default::default());
///
/// // Create a tensor using the global engine
/// let t = Tensor::<Engine, 2>::zeros([128, 128], &device);
/// ```
#[derive(Debug, Default, Clone)]
pub struct Engine;

impl Backend for Engine {
    type Device = Device;

    type FloatTensorPrimitive = EngineTensor;

    // TODO: either allow default dtype generic or remove associated types entirely?
    type FloatElem = f32;

    type IntTensorPrimitive = EngineTensor;

    type IntElem = i32;

    type BoolTensorPrimitive = EngineTensor;

    type BoolElem = u8;

    type QuantizedTensorPrimitive = EngineTensor;

    fn name(device: &Self::Device) -> String {
        let inner = dispatch_device!(device, |device| B::name(device));
        format!("engine<{inner}>")
    }

    fn seed(device: &Self::Device, seed: u64) {
        dispatch_device!(device, |device| B::seed(device, seed))
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        dispatch_device!(device, |device| B::sync(device))
    }

    fn dtype_usage(device: &Self::Device, dtype: DType) -> burn_backend::DTypeUsageSet {
        dispatch_device!(device, |device| B::dtype_usage(device, dtype))
    }
}

impl EngineTensor {
    pub(crate) fn device(&self) -> Device {
        match self {
            #[cfg(feature = "cpu")]
            EngineTensor::Cpu(tensor) => Device::Cpu(tensor.device()),
            #[cfg(feature = "cuda")]
            EngineTensor::Cuda(tensor) => Device::Cuda(tensor.device()),
            #[cfg(feature = "metal")]
            EngineTensor::Metal(tensor) => Device::Metal(tensor.device()),
            #[cfg(feature = "rocm")]
            EngineTensor::Rocm(tensor) => Device::Rocm(tensor.device()),
            #[cfg(feature = "vulkan")]
            EngineTensor::Vulkan(tensor) => Device::Vulkan(tensor.device()),
            #[cfg(feature = "webgpu")]
            EngineTensor::WebGpu(tensor) => Device::WebGpu(tensor.device()),
            #[cfg(feature = "ndarray")]
            EngineTensor::NdArray(tensor) => Device::NdArray(tensor.device()),
            #[cfg(feature = "tch")]
            EngineTensor::LibTorch(tensor) => Device::LibTorch(tensor.device()),
        }
    }
}
