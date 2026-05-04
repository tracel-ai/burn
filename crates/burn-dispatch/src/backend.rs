use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;

use burn_backend::quantization::QuantScheme;
use burn_backend::tensor::{Device, QuantizedTensor};
use burn_backend::{Backend, BackendTypes, DType, ExecutionError, QTensorPrimitive};

#[cfg(feature = "autodiff")]
use burn_autodiff::grads::Gradients;
#[cfg(feature = "autodiff")]
use burn_backend::AutodiffBackend;

#[allow(unused)]
use crate::BackendId;
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

impl BackendTypes for Dispatch {
    type Device = DispatchDevice;

    type FloatTensorPrimitive = DispatchTensor;

    // TODO: either allow default dtype generic or remove associated types entirely?
    type FloatElem = f32;

    type IntTensorPrimitive = DispatchTensor;

    type IntElem = i32;

    type BoolTensorPrimitive = DispatchTensor;

    type BoolElem = u8;

    type QuantizedTensorPrimitive = DispatchTensor;
}

impl Backend for Dispatch {
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

    fn device_count(type_id: u16) -> usize {
        let (dispatch_id, backend_type_id) = DispatchDevice::decode_type_id(type_id);
        match dispatch_id {
            #[cfg(feature = "cpu")]
            BackendId::Cpu => Cpu::<f32>::device_count(backend_type_id),
            #[cfg(feature = "cuda")]
            BackendId::Cuda => Cuda::<f32>::device_count(backend_type_id),
            #[cfg(wgpu_metal)]
            BackendId::Metal => Metal::<f32>::device_count(backend_type_id),
            #[cfg(feature = "rocm")]
            BackendId::Rocm => Rocm::<f32>::device_count(backend_type_id),
            #[cfg(wgpu_vulkan)]
            BackendId::Vulkan => Vulkan::<f32>::device_count(backend_type_id),
            #[cfg(wgpu_webgpu)]
            BackendId::Wgpu => Wgpu::<f32>::device_count(backend_type_id),
            #[cfg(feature = "flex")]
            BackendId::Flex => Flex::device_count(backend_type_id),
            #[cfg(feature = "ndarray")]
            BackendId::NdArray => NdArray::<f32>::device_count(backend_type_id),
            #[cfg(feature = "tch")]
            BackendId::LibTorch => LibTorch::<f32>::device_count(backend_type_id),
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
                DispatchTensorKind::Wgpu(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "flex")]
                DispatchTensorKind::Flex(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "ndarray")]
                DispatchTensorKind::NdArray(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "tch")]
                DispatchTensorKind::LibTorch(tensor) => tensor.autodiff().backward(),
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
                DispatchTensorKind::Wgpu(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Wgpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "flex")]
                DispatchTensorKind::Flex(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Flex(crate::BackendTensor::Float(t))),
                #[cfg(feature = "ndarray")]
                DispatchTensorKind::NdArray(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::NdArray(crate::BackendTensor::Float(t))),
                #[cfg(feature = "tch")]
                DispatchTensorKind::LibTorch(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::LibTorch(crate::BackendTensor::Float(t))),
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
                DispatchTensorKind::Wgpu(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Wgpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "flex")]
                DispatchTensorKind::Flex(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Flex(crate::BackendTensor::Float(t))),
                #[cfg(feature = "ndarray")]
                DispatchTensorKind::NdArray(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::NdArray(crate::BackendTensor::Float(t))),
                #[cfg(feature = "tch")]
                DispatchTensorKind::LibTorch(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::LibTorch(crate::BackendTensor::Float(t))),
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
                (DispatchTensorKind::Wgpu(tensor), DispatchTensorKind::Wgpu(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "flex")]
                (DispatchTensorKind::Flex(tensor), DispatchTensorKind::Flex(grad)) => {
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
                DispatchTensorKind::Wgpu(tensor) => DispatchTensorKind::Wgpu(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(feature = "flex")]
                DispatchTensorKind::Flex(tensor) => DispatchTensorKind::Flex(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(feature = "ndarray")]
                DispatchTensorKind::NdArray(tensor) => DispatchTensorKind::NdArray(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(feature = "tch")]
                DispatchTensorKind::LibTorch(tensor) => DispatchTensorKind::LibTorch(
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
            DispatchTensorKind::Wgpu(tensor) => DispatchTensorKind::Autodiff(Box::new(
                DispatchTensorKind::Wgpu(crate::BackendTensor::Autodiff(
                    Autodiff::<Wgpu<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(feature = "flex")]
            DispatchTensorKind::Flex(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Flex(
                    crate::BackendTensor::Autodiff(Autodiff::<Flex>::from_inner(tensor.float())),
                )))
            }
            #[cfg(feature = "ndarray")]
            DispatchTensorKind::NdArray(tensor) => DispatchTensorKind::Autodiff(Box::new(
                DispatchTensorKind::NdArray(crate::BackendTensor::Autodiff(
                    Autodiff::<NdArray<f32>>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(feature = "tch")]
            DispatchTensorKind::LibTorch(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::LibTorch(
                    crate::BackendTensor::Autodiff(Autodiff::<LibTorch<f32>>::from_inner(
                        tensor.float(),
                    )),
                )))
            }
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
            DispatchTensorKind::Wgpu(tensor) => DispatchDevice::Wgpu(tensor.device()),
            #[cfg(feature = "flex")]
            DispatchTensorKind::Flex(tensor) => DispatchDevice::Flex(tensor.device()),
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
        #[allow(unused_mut)]
        let mut device = self.kind.device();

        #[cfg(feature = "autodiff")]
        if let DispatchDevice::Autodiff(device) = &mut device {
            device.checkpointing = self.checkpointing;
        }

        device
    }
}

impl Dispatch {
    /// Returns the default tensor quantization scheme for the device.
    // TODO: replace this + QTensorPrimitive trait method with better API.
    // This is temporary, for test purposes.
    pub fn default_quant_scheme(device: &Device<Self>) -> QuantScheme {
        match device {
            #[cfg(feature = "cpu")]
            DispatchDevice::Cpu(_) => <QuantizedTensor<Cpu> as QTensorPrimitive>::default_scheme(),
            #[cfg(feature = "cuda")]
            DispatchDevice::Cuda(_) => {
                <QuantizedTensor<Cuda> as QTensorPrimitive>::default_scheme()
            }
            #[cfg(wgpu_metal)]
            DispatchDevice::Metal(_) => {
                <QuantizedTensor<Metal> as QTensorPrimitive>::default_scheme()
            }
            #[cfg(feature = "rocm")]
            DispatchDevice::Rocm(_) => {
                <QuantizedTensor<Rocm> as QTensorPrimitive>::default_scheme()
            }
            #[cfg(wgpu_vulkan)]
            DispatchDevice::Vulkan(_) => {
                <QuantizedTensor<Vulkan> as QTensorPrimitive>::default_scheme()
            }
            #[cfg(wgpu_webgpu)]
            DispatchDevice::Wgpu(_) => {
                <QuantizedTensor<Wgpu> as QTensorPrimitive>::default_scheme()
            }
            #[cfg(feature = "flex")]
            DispatchDevice::Flex(_) => {
                <QuantizedTensor<Flex> as QTensorPrimitive>::default_scheme()
            }
            #[cfg(feature = "ndarray")]
            DispatchDevice::NdArray(_) => {
                <QuantizedTensor<NdArray> as QTensorPrimitive>::default_scheme()
            }
            #[cfg(feature = "tch")]
            DispatchDevice::LibTorch(_) => {
                <QuantizedTensor<LibTorch> as QTensorPrimitive>::default_scheme()
            }
            #[cfg(feature = "autodiff")]
            DispatchDevice::Autodiff(ad_device) => Self::default_quant_scheme(&ad_device.inner),
        }
    }
}
