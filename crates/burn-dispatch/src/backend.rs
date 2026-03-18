use alloc::format;
use alloc::string::String;

use burn_backend::Backend;
use burn_backend::ExecutionError;
#[cfg(feature = "cpu")]
use burn_cpu::Cpu;
#[cfg(feature = "rocm")]
use burn_rocm::Rocm;
use burn_std::DType;

#[cfg(feature = "autodiff")]
use burn_autodiff::grads::Gradients;
#[cfg(feature = "autodiff")]
use burn_backend::AutodiffBackend;
#[cfg(feature = "tch")]
use burn_tch::LibTorch;
#[cfg(wgpu_webgpu)]
use burn_wgpu::Wgpu;
#[cfg(wgpu_metal)]
use burn_wgpu::graphics::Metal;

use crate::DispatchTensorKind;
use crate::backends::*;
use crate::{DispatchDevice, DispatchTensor};

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
/// use burn::DispatchDevice;
///
/// // Select the device to execute operations on
/// let device = DispatchDevice::Cuda(Default::default());
///
/// // Create a tensor using the global backend
/// let t = Tensor::<Dispatch, 2>::zeros([128, 128], &device);
/// ```
#[derive(Debug, Default, Clone)]
pub struct Dispatch;

impl Backend for Dispatch {
    type Device = DispatchDevice;

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
            DispatchDevice::Autodiff(_) => true,
            _ => false,
        }
    }
}

#[cfg(feature = "autodiff")]
impl AutodiffBackend for Dispatch {
    type InnerBackend = Dispatch;

    type Gradients = Gradients;

    fn backward(tensor: DispatchTensor) -> Self::Gradients {
        let DispatchTensor { kind, .. } = tensor;
        match kind {
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(tensor) => match *tensor {
                #[cfg(feature = "cpu")]
                DispatchTensorKind::Cpu(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "cuda")]
                DispatchTensorKind::Cuda(tensor) => tensor.autodiff().backward(),
                #[cfg(wgpu_metal)]
                DispatchTensorKind::Metal(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "rocm")]
                DispatchTensorKind::Rocm(tensor) => tensor.autodiff().backward(),
                #[cfg(wgpu_vulkan)]
                DispatchTensorKind::Vulkan(tensor) => tensor.autodiff().backward(),
                #[cfg(wgpu_webgpu)]
                DispatchTensorKind::WebGpu(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "ndarray")]
                DispatchTensorKind::NdArray(tensor) => tensor.autodiff().backward(),
                DispatchTensorKind::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn grad(tensor: &DispatchTensor, grads: &Self::Gradients) -> Option<DispatchTensor> {
        let DispatchTensor {
            kind,
            checkpointing,
        } = tensor;
        let grad = match &kind {
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(inner_kind) => match &**inner_kind {
                #[cfg(feature = "cpu")]
                DispatchTensorKind::Cpu(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Cpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "cuda")]
                DispatchTensorKind::Cuda(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Cuda(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_metal)]
                DispatchTensorKind::Metal(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Metal(crate::BackendTensor::Float(t))),
                #[cfg(feature = "rocm")]
                DispatchTensorKind::Rocm(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Rocm(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_vulkan)]
                DispatchTensorKind::Vulkan(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Vulkan(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_webgpu)]
                DispatchTensorKind::WebGpu(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::WebGpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "ndarray")]
                DispatchTensorKind::NdArray(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::NdArray(crate::BackendTensor::Float(t))),
                DispatchTensorKind::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        };
        grad.map(|kind| DispatchTensor {
            kind,
            checkpointing: *checkpointing,
        })
    }

    fn grad_remove(tensor: &DispatchTensor, grads: &mut Self::Gradients) -> Option<DispatchTensor> {
        let DispatchTensor {
            kind,
            checkpointing,
        } = tensor;
        let grad = match &kind {
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(inner_kind) => match &**inner_kind {
                #[cfg(feature = "cpu")]
                DispatchTensorKind::Cpu(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Cpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "cuda")]
                DispatchTensorKind::Cuda(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Cuda(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_metal)]
                DispatchTensorKind::Metal(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Metal(crate::BackendTensor::Float(t))),
                #[cfg(feature = "rocm")]
                DispatchTensorKind::Rocm(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Rocm(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_vulkan)]
                DispatchTensorKind::Vulkan(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Vulkan(crate::BackendTensor::Float(t))),
                #[cfg(wgpu_webgpu)]
                DispatchTensorKind::WebGpu(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::WebGpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "ndarray")]
                DispatchTensorKind::NdArray(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::NdArray(crate::BackendTensor::Float(t))),
                DispatchTensorKind::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        };
        grad.map(|kind| DispatchTensor {
            kind,
            checkpointing: *checkpointing,
        })
    }

    fn grad_replace(tensor: &DispatchTensor, grads: &mut Self::Gradients, grad: DispatchTensor) {
        let DispatchTensor {
            kind,
            checkpointing,
        } = tensor;
        let DispatchTensor {
            kind: grad,
            checkpointing: grad_ckp,
        } = grad;
        debug_assert_eq!(checkpointing, &grad_ckp);

        match &kind {
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(inner_kind) => match (&**inner_kind, grad) {
                #[cfg(feature = "cpu")]
                (DispatchTensorKind::Cpu(tensor), DispatchTensorKind::Cpu(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "cuda")]
                (DispatchTensorKind::Cuda(tensor), DispatchTensorKind::Cuda(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(wgpu_metal)]
                (DispatchTensorKind::Metal(tensor), DispatchTensorKind::Metal(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "rocm")]
                (DispatchTensorKind::Rocm(tensor), DispatchTensorKind::Rocm(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(wgpu_vulkan)]
                (DispatchTensorKind::Vulkan(tensor), DispatchTensorKind::Vulkan(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(wgpu_webgpu)]
                (DispatchTensorKind::WebGpu(tensor), DispatchTensorKind::WebGpu(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "ndarray")]
                (DispatchTensorKind::NdArray(tensor), DispatchTensorKind::NdArray(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                (DispatchTensorKind::Autodiff(_), _) => {
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
        let DispatchTensor {
            kind,
            checkpointing,
        } = tensor;

        let kind = match kind {
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(inner_kind) => match *inner_kind {
                #[cfg(feature = "cpu")]
                DispatchTensorKind::Cpu(tensor) => DispatchTensorKind::Cpu(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(feature = "cuda")]
                DispatchTensorKind::Cuda(tensor) => DispatchTensorKind::Cuda(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(wgpu_metal)]
                DispatchTensorKind::Metal(tensor) => DispatchTensorKind::Metal(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(feature = "rocm")]
                DispatchTensorKind::Rocm(tensor) => DispatchTensorKind::Rocm(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(wgpu_vulkan)]
                DispatchTensorKind::Vulkan(tensor) => DispatchTensorKind::Vulkan(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(wgpu_webgpu)]
                DispatchTensorKind::WebGpu(tensor) => DispatchTensorKind::WebGpu(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(feature = "ndarray")]
                DispatchTensorKind::NdArray(tensor) => DispatchTensorKind::NdArray(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                DispatchTensorKind::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        };
        DispatchTensor {
            kind,
            checkpointing,
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
        let DispatchTensor {
            kind,
            checkpointing,
        } = tensor;

        let kind = match kind {
            #[cfg(feature = "cpu")]
            DispatchTensorKind::Cpu(tensor) => DispatchTensorKind::Autodiff(Box::new(
                DispatchTensorKind::Cpu(crate::BackendTensor::Autodiff(
                    Autodiff::<Cpu<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(feature = "cuda")]
            DispatchTensorKind::Cuda(tensor) => DispatchTensorKind::Autodiff(Box::new(
                DispatchTensorKind::Cuda(crate::BackendTensor::Autodiff(
                    Autodiff::<Cuda<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(wgpu_metal)]
            DispatchTensorKind::Metal(tensor) => DispatchTensorKind::Autodiff(Box::new(
                DispatchTensorKind::Metal(crate::BackendTensor::Autodiff(
                    Autodiff::<Metal<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(feature = "rocm")]
            DispatchTensorKind::Rocm(tensor) => DispatchTensorKind::Autodiff(Box::new(
                DispatchTensorKind::Rocm(crate::BackendTensor::Autodiff(
                    Autodiff::<Rocm<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(wgpu_vulkan)]
            DispatchTensorKind::Vulkan(tensor) => DispatchTensorKind::Autodiff(Box::new(
                DispatchTensorKind::Vulkan(crate::BackendTensor::Autodiff(
                    Autodiff::<Vulkan<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(wgpu_webgpu)]
            DispatchTensorKind::WebGpu(tensor) => DispatchTensorKind::Autodiff(Box::new(
                DispatchTensorKind::WebGpu(crate::BackendTensor::Autodiff(
                    Autodiff::<WebGpu<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(feature = "ndarray")]
            DispatchTensorKind::NdArray(tensor) => DispatchTensorKind::Autodiff(Box::new(
                DispatchTensorKind::NdArray(crate::BackendTensor::Autodiff(
                    Autodiff::<NdArray<f32>>::from_inner(tensor.float()),
                )),
            )),
            DispatchTensorKind::Autodiff(_) => {
                panic!("Autodiff should not wrap an autodiff tensor.")
            }
        };
        DispatchTensor {
            kind,
            checkpointing,
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

impl DispatchTensorKind {
    pub(crate) fn device(&self) -> DispatchDevice {
        match self {
            #[cfg(feature = "cpu")]
            DispatchTensorKind::Cpu(tensor) => DispatchDevice::Cpu(tensor.device()),
            #[cfg(feature = "cuda")]
            DispatchTensorKind::Cuda(tensor) => DispatchDevice::Cuda(tensor.device()),
            #[cfg(wgpu_metal)]
            DispatchTensorKind::Metal(tensor) => DispatchDevice::Metal(tensor.device()),
            #[cfg(feature = "rocm")]
            DispatchTensorKind::Rocm(tensor) => DispatchDevice::Rocm(tensor.device()),
            #[cfg(wgpu_vulkan)]
            DispatchTensorKind::Vulkan(tensor) => DispatchDevice::Vulkan(tensor.device()),
            #[cfg(wgpu_webgpu)]
            DispatchTensorKind::WebGpu(tensor) => DispatchDevice::WebGpu(tensor.device()),
            #[cfg(feature = "ndarray")]
            DispatchTensorKind::NdArray(tensor) => DispatchDevice::NdArray(tensor.device()),
            #[cfg(feature = "tch")]
            DispatchTensorKind::LibTorch(tensor) => DispatchDevice::LibTorch(tensor.device()),
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(tensor) => DispatchDevice::autodiff(tensor.device()),
        }
    }
}

impl DispatchTensor {
    pub(crate) fn device(&self) -> DispatchDevice {
        self.kind.device()
    }
}
