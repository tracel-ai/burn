use alloc::format;
use alloc::string::String;

use burn_backend::Backend;
use burn_backend::ExecutionError;
use burn_std::DType;

#[cfg(feature = "autodiff")]
use burn_autodiff::grads::Gradients;
#[cfg(feature = "autodiff")]
use burn_backend::AutodiffBackend;

use crate::backends::*;
use crate::{Device, DispatchTensor};

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

    fn ad_enabled(device: &Self::Device) -> bool {
        match device {
            #[cfg(feature = "autodiff")]
            Device::Autodiff(_) => true,
            _ => false,
        }
    }
}

#[cfg(feature = "autodiff")]
impl AutodiffBackend for Dispatch {
    type InnerBackend = Dispatch;

    type Gradients = Gradients;

    fn backward(tensor: DispatchTensor) -> Self::Gradients {
        match tensor {
            #[cfg(feature = "autodiff")]
            DispatchTensor::Autodiff(tensor) => match *tensor {
                #[cfg(feature = "cpu")]
                DispatchTensor::Cpu(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "cuda")]
                DispatchTensor::Cuda(tensor) => tensor.autodiff().backward(),
                #[cfg(wgpu_metal)]
                DispatchTensor::Metal(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "rocm")]
                DispatchTensor::Rocm(tensor) => tensor.autodiff().backward(),
                #[cfg(wgpu_vulkan)]
                DispatchTensor::Vulkan(tensor) => tensor.autodiff().backward(),
                #[cfg(wgpu_webgpu)]
                DispatchTensor::WebGpu(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "ndarray")]
                DispatchTensor::NdArray(tensor) => tensor.autodiff().backward(),
                DispatchTensor::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn grad(tensor: &DispatchTensor, grads: &Self::Gradients) -> Option<DispatchTensor> {
        match &tensor {
            #[cfg(feature = "autodiff")]
            DispatchTensor::Autodiff(tensor) => match &**tensor {
                #[cfg(feature = "cpu")]
                DispatchTensor::Cpu(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensor::Cpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "cuda")]
                DispatchTensor::Cuda(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensor::Cuda(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_metal)]
                DispatchTensor::Metal(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensor::Metal(crate::BackendTensor::Float(t))),
                #[cfg(feature = "rocm")]
                DispatchTensor::Rocm(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensor::Rocm(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_vulkan)]
                DispatchTensor::Vulkan(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensor::Vulkan(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_webgpu)]
                DispatchTensor::WebGpu(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensor::WebGpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "ndarray")]
                DispatchTensor::NdArray(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensor::NdArray(crate::BackendTensor::Float(t))),
                DispatchTensor::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn grad_remove(tensor: &DispatchTensor, grads: &mut Self::Gradients) -> Option<DispatchTensor> {
        match &tensor {
            #[cfg(feature = "autodiff")]
            DispatchTensor::Autodiff(tensor) => match &**tensor {
                #[cfg(feature = "cpu")]
                DispatchTensor::Cpu(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensor::Cpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "cuda")]
                DispatchTensor::Cuda(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensor::Cuda(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_metal)]
                DispatchTensor::Metal(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensor::Metal(crate::BackendTensor::Float(t))),
                #[cfg(feature = "rocm")]
                DispatchTensor::Rocm(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensor::Rocm(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_vulkan)]
                DispatchTensor::Vulkan(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensor::Vulkan(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_webgpu)]
                DispatchTensor::WebGpu(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensor::WebGpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "ndarray")]
                DispatchTensor::NdArray(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensor::NdArray(crate::BackendTensor::Float(t))),
                DispatchTensor::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn grad_replace(tensor: &DispatchTensor, grads: &mut Self::Gradients, grad: DispatchTensor) {
        match &tensor {
            #[cfg(feature = "autodiff")]
            DispatchTensor::Autodiff(tensor) => match (&**tensor, grad) {
                #[cfg(feature = "cpu")]
                (DispatchTensor::Cpu(tensor), DispatchTensor::Cpu(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "cuda")]
                (DispatchTensor::Cuda(tensor), DispatchTensor::Cuda(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(wgpu_metal)]
                (DispatchTensor::Metal(tensor), DispatchTensor::Metal(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "rocm")]
                (DispatchTensor::Rocm(tensor), DispatchTensor::Rocm(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(wgpu_vulkan)]
                (DispatchTensor::Vulkan(tensor), DispatchTensor::Vulkan(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(wgpu_webgpu)]
                (DispatchTensor::WebGpu(tensor), DispatchTensor::WebGpu(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "ndarray")]
                (DispatchTensor::NdArray(tensor), DispatchTensor::NdArray(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                (DispatchTensor::Autodiff(_), _) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
                (t, g) => panic!(
                    "The provided tensors are not on the same backend. Got backends {t:?} and {g:?}."
                ),
            },
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn inner(tensor: DispatchTensor) -> DispatchTensor {
        match tensor {
            #[cfg(feature = "autodiff")]
            DispatchTensor::Autodiff(tensor) => match *tensor {
                #[cfg(feature = "cpu")]
                DispatchTensor::Cpu(tensor) => {
                    DispatchTensor::Cpu(crate::BackendTensor::Float(tensor.autodiff().primitive))
                }
                #[cfg(feature = "cuda")]
                DispatchTensor::Cuda(tensor) => {
                    DispatchTensor::Cuda(crate::BackendTensor::Float(tensor.autodiff().primitive))
                }
                #[cfg(wgpu_metal)]
                DispatchTensor::Metal(tensor) => {
                    DispatchTensor::Metal(crate::BackendTensor::Float(tensor.autodiff().primitive))
                }
                #[cfg(feature = "rocm")]
                DispatchTensor::Rocm(tensor) => {
                    DispatchTensor::Rocm(crate::BackendTensor::Float(tensor.autodiff().primitive))
                }
                #[cfg(wgpu_vulkan)]
                DispatchTensor::Vulkan(tensor) => {
                    DispatchTensor::Vulkan(crate::BackendTensor::Float(tensor.autodiff().primitive))
                }
                #[cfg(wgpu_webgpu)]
                DispatchTensor::WebGpu(tensor) => {
                    DispatchTensor::WebGpu(crate::BackendTensor::Float(tensor.autodiff().primitive))
                }
                #[cfg(feature = "ndarray")]
                DispatchTensor::NdArray(tensor) => DispatchTensor::NdArray(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                DispatchTensor::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn int_inner(tensor: DispatchTensor) -> DispatchTensor {
        tensor
    }

    fn bool_inner(tensor: DispatchTensor) -> DispatchTensor {
        tensor
    }

    fn q_inner(tensor: DispatchTensor) -> DispatchTensor {
        tensor
    }

    fn from_inner(tensor: DispatchTensor) -> DispatchTensor {
        match tensor {
            #[cfg(feature = "cpu")]
            DispatchTensor::Cpu(tensor) => DispatchTensor::Autodiff(Box::new(DispatchTensor::Cpu(
                crate::BackendTensor::Autodiff(Autodiff::<Cpu<f32>>::from_inner(tensor.float())),
            ))),
            #[cfg(feature = "cuda")]
            DispatchTensor::Cuda(tensor) => DispatchTensor::Autodiff(Box::new(
                DispatchTensor::Cuda(crate::BackendTensor::Autodiff(
                    Autodiff::<Cuda<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(wgpu_metal)]
            DispatchTensor::Metal(tensor) => DispatchTensor::Autodiff(Box::new(
                DispatchTensor::Metal(crate::BackendTensor::Autodiff(
                    Autodiff::<Metal<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(feature = "rocm")]
            DispatchTensor::Rocm(tensor) => DispatchTensor::Autodiff(Box::new(
                DispatchTensor::Rocm(crate::BackendTensor::Autodiff(
                    Autodiff::<Rocm<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(wgpu_vulkan)]
            DispatchTensor::Vulkan(tensor) => DispatchTensor::Autodiff(Box::new(
                DispatchTensor::Vulkan(crate::BackendTensor::Autodiff(
                    Autodiff::<Vulkan<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(wgpu_webgpu)]
            DispatchTensor::WebGpu(tensor) => DispatchTensor::Autodiff(Box::new(
                DispatchTensor::WebGpu(crate::BackendTensor::Autodiff(
                    Autodiff::<WebGpu<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(feature = "ndarray")]
            DispatchTensor::NdArray(tensor) => DispatchTensor::Autodiff(Box::new(
                DispatchTensor::NdArray(crate::BackendTensor::Autodiff(
                    Autodiff::<NdArray<f32>>::from_inner(tensor.float()),
                )),
            )),
            DispatchTensor::Autodiff(_) => {
                panic!("Autodiff should not wrap an autodiff tensor.")
            }
        }
    }

    fn int_from_inner(tensor: DispatchTensor) -> DispatchTensor {
        tensor
    }

    fn bool_from_inner(tensor: DispatchTensor) -> DispatchTensor {
        tensor
    }

    fn q_from_inner(tensor: DispatchTensor) -> DispatchTensor {
        tensor
    }
}

impl DispatchTensor {
    pub(crate) fn device(&self) -> Device {
        match self {
            #[cfg(feature = "cpu")]
            DispatchTensor::Cpu(tensor) => Device::Cpu(tensor.device()),
            #[cfg(feature = "cuda")]
            DispatchTensor::Cuda(tensor) => Device::Cuda(tensor.device()),
            #[cfg(wgpu_metal)]
            DispatchTensor::Metal(tensor) => Device::Metal(tensor.device()),
            #[cfg(feature = "rocm")]
            DispatchTensor::Rocm(tensor) => Device::Rocm(tensor.device()),
            #[cfg(wgpu_vulkan)]
            DispatchTensor::Vulkan(tensor) => Device::Vulkan(tensor.device()),
            #[cfg(wgpu_webgpu)]
            DispatchTensor::WebGpu(tensor) => Device::WebGpu(tensor.device()),
            #[cfg(feature = "ndarray")]
            DispatchTensor::NdArray(tensor) => Device::NdArray(tensor.device()),
            #[cfg(feature = "tch")]
            DispatchTensor::LibTorch(tensor) => Device::LibTorch(tensor.device()),
            #[cfg(feature = "autodiff")]
            DispatchTensor::Autodiff(tensor) => Device::Autodiff(Box::new(tensor.device())),
        }
    }
}
