use alloc::format;
use alloc::string::String;

use burn_backend::Backend;
use burn_backend::ExecutionError;
use burn_std::DType;

use crate::backends::*;
use crate::{Device, DispatchTensor, dispatch_device};

/// The main execution backend in Burn.
///
/// [`Dispatch`] acts as a global backend that can manage multiple underlying
/// backends (e.g., `Cpu`, `Cuda`, `Wgpu`, `Metal`, etc.).  
/// It is responsible for:
/// - Dispatching tensor operations to the appropriate backend.
/// - Managing cross-backend tensor transfers.
///
/// Essentially, [`Dispatch`] is the single entry point for executing tensor operations
/// in a backend-agnostic way. It allows Burn to provide a unified, global backend
/// for users while still leveraging multiple specialized backends under the hood.
///
/// # Example
///
/// ```ignore
/// use burn::Dispatch;
/// use burn::Device;
///
/// // Select the device to execute operations on
/// let device = Device::Cuda(Default::default());
///
/// // Create a tensor using the global backend
/// let t = Tensor::<Dispatch, 2>::zeros([128, 128], &device);
/// ```
#[derive(Debug, Default, Clone)]
pub struct Dispatch;

impl Backend for Dispatch {
    type Device = Device;

    type FloatTensorPrimitive = DispatchTensor;

    // TODO: either allow default dtype generic or remove associated types entirely?
    type FloatElem = f32;

    type IntTensorPrimitive = DispatchTensor;

    type IntElem = i32;

    type BoolTensorPrimitive = DispatchTensor;

    type BoolElem = u8;

    type QuantizedTensorPrimitive = DispatchTensor;

    fn name(device: &Self::Device) -> String {
        let inner = dispatch_device!(device, |device| B::name(device));
        format!("dispatch<{inner}>")
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

impl DispatchTensor {
    pub(crate) fn device(&self) -> Device {
        match self {
            #[cfg(feature = "cpu")]
            DispatchTensor::Cpu(tensor) => Device::Cpu(tensor.device()),
            #[cfg(feature = "cuda")]
            DispatchTensor::Cuda(tensor) => Device::Cuda(tensor.device()),
            #[cfg(feature = "metal")]
            DispatchTensor::Metal(tensor) => Device::Metal(tensor.device()),
            #[cfg(feature = "rocm")]
            DispatchTensor::Rocm(tensor) => Device::Rocm(tensor.device()),
            #[cfg(feature = "vulkan")]
            DispatchTensor::Vulkan(tensor) => Device::Vulkan(tensor.device()),
            #[cfg(feature = "webgpu")]
            DispatchTensor::WebGpu(tensor) => Device::WebGpu(tensor.device()),
            #[cfg(feature = "ndarray")]
            DispatchTensor::NdArray(tensor) => Device::NdArray(tensor.device()),
            #[cfg(feature = "tch")]
            DispatchTensor::LibTorch(tensor) => Device::LibTorch(tensor.device()),
        }
    }
}
