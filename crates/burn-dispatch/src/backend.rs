use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(any(
    feature = "cpu",
    feature = "ndarray",
    feature = "flex",
    default_backend
))]
use alloc::vec;

use burn_backend::{AutodiffBackend, Backend, BackendTypes, DType, ExecutionError};

#[cfg(feature = "autodiff")]
use alloc::boxed::Box;
#[cfg(feature = "autodiff")]
use burn_autodiff::grads::Gradients;

#[allow(unused)]
use crate::DispatchDeviceId;
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
    type IntTensorPrimitive = DispatchTensor;
    type BoolTensorPrimitive = DispatchTensor;
    type QuantizedTensorPrimitive = DispatchTensor;
    type ComplexTensorPrimitive = DispatchTensor;

    fn device_count(type_id: u16) -> usize {
        let (dispatch_id, backend_type_id) = DispatchDevice::decode_type_id(type_id);
        match dispatch_id {
            #[cfg(feature = "cpu")]
            DispatchDeviceId::Cpu => Cpu::device_count(backend_type_id),
            #[cfg(feature = "cuda")]
            DispatchDeviceId::Cuda => Cuda::device_count(backend_type_id),
            #[cfg(feature = "metal")]
            DispatchDeviceId::Metal => Metal::device_count(backend_type_id),
            #[cfg(feature = "rocm")]
            DispatchDeviceId::Rocm => Rocm::device_count(backend_type_id),
            #[cfg(feature = "vulkan")]
            DispatchDeviceId::Vulkan => Vulkan::device_count(backend_type_id),
            #[cfg(feature = "wgpu")]
            DispatchDeviceId::Wgpu => Wgpu::device_count(backend_type_id),
            #[cfg(feature = "webgpu")]
            DispatchDeviceId::WebGpu => WebGpu::device_count(backend_type_id),
            #[cfg(feature = "flex")]
            DispatchDeviceId::Flex => Flex::device_count(backend_type_id),
            #[cfg(any(feature = "ndarray", default_backend))]
            DispatchDeviceId::NdArray => NdArray::device_count(backend_type_id),
            #[cfg(feature = "tch")]
            DispatchDeviceId::LibTorch => LibTorch::device_count(backend_type_id),
            #[cfg(feature = "remote")]
            DispatchDeviceId::Remote => Remote::device_count(backend_type_id),
            _ => unreachable!("No backend feature enabled."),
        }
    }
    fn dtype_usage(device: &Self::Device, dtype: DType) -> burn_backend::DTypeUsageSet {
        dispatch_device!(device, |device| B::dtype_usage(device, dtype))
    }
    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool {
        dispatch_device!(device, |device| B::supports_dtype(device, dtype))
    }
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

    fn ad_enabled(device: &Self::Device) -> bool {
        match device {
            #[cfg(feature = "autodiff")]
            DispatchDevice::Autodiff(_) => true,
            _ => false,
        }
    }

    fn memory_persistent_allocations<
        Output: Send,
        Input: Send,
        Func: Fn(Input) -> Output + Send,
    >(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        dispatch_device!(device, |device| B::memory_persistent_allocations(
            device, input, func
        ))
    }

    fn memory_cleanup(device: &Self::Device) {
        dispatch_device!(device, |device| B::memory_cleanup(device))
    }

    fn staging<'a, Iter>(data: Iter, device: &Self::Device)
    where
        Iter: Iterator<Item = &'a mut burn_backend::TensorData>,
    {
        dispatch_device!(device, |device| B::staging(data, device))
    }
}

#[cfg(feature = "autodiff")]
impl AutodiffBackend for Dispatch {
    type InnerBackend = Dispatch;

    type Gradients = Gradients;

    fn backward(tensor: DispatchTensor) -> Self::Gradients {
        let DispatchTensor { kind, .. } = tensor;
        match kind {
            DispatchTensorKind::Autodiff(tensor) => match *tensor {
                #[cfg(feature = "cpu")]
                DispatchTensorKind::Cpu(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "cuda")]
                DispatchTensorKind::Cuda(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "metal")]
                DispatchTensorKind::Metal(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "rocm")]
                DispatchTensorKind::Rocm(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "vulkan")]
                DispatchTensorKind::Vulkan(tensor) => tensor.autodiff_float().backward(),
                #[cfg(feature = "wgpu")]
                DispatchTensorKind::Wgpu(tensor) => tensor.autodiff_float().backward(),
                #[cfg(feature = "webgpu")]
                DispatchTensorKind::WebGpu(tensor) => tensor.autodiff_float().backward(),
                #[cfg(feature = "flex")]
                DispatchTensorKind::Flex(tensor) => tensor.autodiff_float().backward(),
                #[cfg(any(feature = "ndarray", default_backend))]
                DispatchTensorKind::NdArray(tensor) => tensor.autodiff_float().backward(),
                #[cfg(feature = "tch")]
                DispatchTensorKind::LibTorch(tensor) => tensor.autodiff_float().backward(),
                #[cfg(feature = "remote")]
                DispatchTensorKind::Remote(tensor) => tensor.autodiff_float().backward(),
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
            DispatchTensorKind::Autodiff(inner_kind) => match &**inner_kind {
                #[cfg(feature = "cpu")]
                DispatchTensorKind::Cpu(tensor) => tensor
                    .as_autodiff_float()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Cpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "cuda")]
                DispatchTensorKind::Cuda(tensor) => tensor
                    .as_autodiff_float()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Cuda(crate::BackendTensor::Float(t))),
                #[cfg(feature = "metal")]
                DispatchTensorKind::Metal(tensor) => tensor
                    .as_autodiff_float()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Metal(crate::BackendTensor::Float(t))),
                #[cfg(feature = "rocm")]
                DispatchTensorKind::Rocm(tensor) => tensor
                    .as_autodiff_float()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Rocm(crate::BackendTensor::Float(t))),
                #[cfg(feature = "vulkan")]
                DispatchTensorKind::Vulkan(tensor) => tensor
                    .as_autodiff_float()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Vulkan(crate::BackendTensor::Float(t))),
                #[cfg(feature = "wgpu")]
                DispatchTensorKind::Wgpu(tensor) => tensor
                    .as_autodiff_float()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Wgpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "webgpu")]
                DispatchTensorKind::WebGpu(tensor) => tensor
                    .as_autodiff_float()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::WebGpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "flex")]
                DispatchTensorKind::Flex(tensor) => tensor
                    .as_autodiff_float()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Flex(crate::BackendTensor::Float(t))),
                #[cfg(any(feature = "ndarray", default_backend))]
                DispatchTensorKind::NdArray(tensor) => tensor
                    .as_autodiff_float()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::NdArray(crate::BackendTensor::Float(t))),
                #[cfg(feature = "tch")]
                DispatchTensorKind::LibTorch(tensor) => tensor
                    .as_autodiff_float()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::LibTorch(crate::BackendTensor::Float(t))),
                #[cfg(feature = "remote")]
                DispatchTensorKind::Remote(tensor) => tensor
                    .as_autodiff_float()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Remote(crate::BackendTensor::Float(t))),
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
            DispatchTensorKind::Autodiff(inner_kind) => match &**inner_kind {
                #[cfg(feature = "cpu")]
                DispatchTensorKind::Cpu(tensor) => tensor
                    .as_autodiff_float()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Cpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "cuda")]
                DispatchTensorKind::Cuda(tensor) => tensor
                    .as_autodiff_float()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Cuda(crate::BackendTensor::Float(t))),
                #[cfg(feature = "metal")]
                DispatchTensorKind::Metal(tensor) => tensor
                    .as_autodiff_float()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Metal(crate::BackendTensor::Float(t))),
                #[cfg(feature = "rocm")]
                DispatchTensorKind::Rocm(tensor) => tensor
                    .as_autodiff_float()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Rocm(crate::BackendTensor::Float(t))),
                #[cfg(feature = "vulkan")]
                DispatchTensorKind::Vulkan(tensor) => tensor
                    .as_autodiff_float()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Vulkan(crate::BackendTensor::Float(t))),
                #[cfg(feature = "wgpu")]
                DispatchTensorKind::Wgpu(tensor) => tensor
                    .as_autodiff_float()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Wgpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "webgpu")]
                DispatchTensorKind::WebGpu(tensor) => tensor
                    .as_autodiff_float()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::WebGpu(crate::BackendTensor::Float(t))),
                #[cfg(feature = "flex")]
                DispatchTensorKind::Flex(tensor) => tensor
                    .as_autodiff_float()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Flex(crate::BackendTensor::Float(t))),
                #[cfg(any(feature = "ndarray", default_backend))]
                DispatchTensorKind::NdArray(tensor) => tensor
                    .as_autodiff_float()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::NdArray(crate::BackendTensor::Float(t))),
                #[cfg(feature = "tch")]
                DispatchTensorKind::LibTorch(tensor) => tensor
                    .as_autodiff_float()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::LibTorch(crate::BackendTensor::Float(t))),
                #[cfg(feature = "remote")]
                DispatchTensorKind::Remote(tensor) => tensor
                    .as_autodiff_float()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Remote(crate::BackendTensor::Float(t))),
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
            DispatchTensorKind::Autodiff(inner_kind) => match (&**inner_kind, grad) {
                #[cfg(feature = "cpu")]
                (DispatchTensorKind::Cpu(tensor), DispatchTensorKind::Cpu(grad)) => {
                    tensor.as_autodiff_float().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "cuda")]
                (DispatchTensorKind::Cuda(tensor), DispatchTensorKind::Cuda(grad)) => {
                    tensor.as_autodiff_float().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "metal")]
                (DispatchTensorKind::Metal(tensor), DispatchTensorKind::Metal(grad)) => {
                    tensor.as_autodiff_float().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "rocm")]
                (DispatchTensorKind::Rocm(tensor), DispatchTensorKind::Rocm(grad)) => {
                    tensor.as_autodiff_float().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "vulkan")]
                (DispatchTensorKind::Vulkan(tensor), DispatchTensorKind::Vulkan(grad)) => {
                    tensor.as_autodiff_float().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "wgpu")]
                (DispatchTensorKind::Wgpu(tensor), DispatchTensorKind::Wgpu(grad)) => {
                    tensor.as_autodiff_float().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "webgpu")]
                (DispatchTensorKind::WebGpu(tensor), DispatchTensorKind::WebGpu(grad)) => {
                    tensor.as_autodiff_float().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "flex")]
                (DispatchTensorKind::Flex(tensor), DispatchTensorKind::Flex(grad)) => {
                    tensor.as_autodiff_float().grad_replace(grads, grad.float())
                }
                #[cfg(any(feature = "ndarray", default_backend))]
                (DispatchTensorKind::NdArray(tensor), DispatchTensorKind::NdArray(grad)) => {
                    tensor.as_autodiff_float().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "remote")]
                (DispatchTensorKind::Remote(tensor), DispatchTensorKind::Remote(grad)) => {
                    tensor.as_autodiff_float().grad_replace(grads, grad.float())
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
            DispatchTensorKind::Autodiff(inner_kind) => match *inner_kind {
                #[cfg(feature = "cpu")]
                DispatchTensorKind::Cpu(tensor) => DispatchTensorKind::Cpu(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(feature = "cuda")]
                DispatchTensorKind::Cuda(tensor) => DispatchTensorKind::Cuda(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(feature = "metal")]
                DispatchTensorKind::Metal(tensor) => DispatchTensorKind::Metal(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(feature = "rocm")]
                DispatchTensorKind::Rocm(tensor) => DispatchTensorKind::Rocm(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(feature = "vulkan")]
                DispatchTensorKind::Vulkan(tensor) => DispatchTensorKind::Vulkan(
                    crate::BackendTensor::Float(tensor.autodiff_float().primitive),
                ),
                #[cfg(feature = "wgpu")]
                DispatchTensorKind::Wgpu(tensor) => DispatchTensorKind::Wgpu(
                    crate::BackendTensor::Float(tensor.autodiff_float().primitive),
                ),
                #[cfg(feature = "webgpu")]
                DispatchTensorKind::WebGpu(tensor) => DispatchTensorKind::WebGpu(
                    crate::BackendTensor::Float(tensor.autodiff_float().primitive),
                ),
                #[cfg(feature = "flex")]
                DispatchTensorKind::Flex(tensor) => DispatchTensorKind::Flex(
                    crate::BackendTensor::Float(tensor.autodiff_float().primitive),
                ),
                #[cfg(any(feature = "ndarray", default_backend))]
                DispatchTensorKind::NdArray(tensor) => DispatchTensorKind::NdArray(
                    crate::BackendTensor::Float(tensor.autodiff_float().primitive),
                ),
                #[cfg(feature = "tch")]
                DispatchTensorKind::LibTorch(tensor) => DispatchTensorKind::LibTorch(
                    crate::BackendTensor::Float(tensor.autodiff_float().primitive),
                ),
                #[cfg(feature = "remote")]
                DispatchTensorKind::Remote(tensor) => DispatchTensorKind::Remote(
                    crate::BackendTensor::Float(tensor.autodiff_float().primitive),
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
            DispatchTensorKind::Cpu(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Cpu(
                    crate::BackendTensor::Autodiff(Autodiff::<Cpu>::from_inner(tensor.float())),
                )))
            }
            #[cfg(feature = "cuda")]
            DispatchTensorKind::Cuda(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Cuda(
                    crate::BackendTensor::Autodiff(Autodiff::<Cuda>::from_inner(tensor.float())),
                )))
            }
            #[cfg(feature = "metal")]
            DispatchTensorKind::Metal(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Metal(
                    crate::BackendTensor::Autodiff(Autodiff::<Metal>::from_inner(tensor.float())),
                )))
            }
            #[cfg(feature = "rocm")]
            DispatchTensorKind::Rocm(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Rocm(
                    crate::BackendTensor::Autodiff(Autodiff::<Rocm>::from_inner(tensor.float())),
                )))
            }
            #[cfg(feature = "vulkan")]
            DispatchTensorKind::Vulkan(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Vulkan(
                    crate::BackendTensor::Autodiff(Autodiff::<Vulkan>::from_inner(tensor.float())),
                )))
            }
            #[cfg(feature = "wgpu")]
            DispatchTensorKind::Wgpu(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Wgpu(
                    crate::BackendTensor::Autodiff(Autodiff::<Wgpu>::from_inner(tensor.float())),
                )))
            }
            #[cfg(feature = "webgpu")]
            DispatchTensorKind::WebGpu(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::WebGpu(
                    crate::BackendTensor::Autodiff(Autodiff::<WebGpu>::from_inner(tensor.float())),
                )))
            }
            #[cfg(feature = "flex")]
            DispatchTensorKind::Flex(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Flex(
                    crate::BackendTensor::Autodiff(Autodiff::<Flex>::from_inner(tensor.float())),
                )))
            }
            #[cfg(any(feature = "ndarray", default_backend))]
            DispatchTensorKind::NdArray(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::NdArray(
                    crate::BackendTensor::Autodiff(Autodiff::<NdArray>::from_inner(tensor.float())),
                )))
            }
            #[cfg(feature = "tch")]
            DispatchTensorKind::LibTorch(tensor) => DispatchTensorKind::Autodiff(Box::new(
                DispatchTensorKind::LibTorch(crate::BackendTensor::Autodiff(
                    Autodiff::<LibTorch>::from_inner(tensor.float()),
                )),
            )),
            #[cfg(feature = "remote")]
            DispatchTensorKind::Remote(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Remote(
                    crate::BackendTensor::Autodiff(Autodiff::<Remote>::from_inner(tensor.float())),
                )))
            }
            DispatchTensorKind::Autodiff(_) => {
                panic!("Autodiff should not wrap an autodiff tensor.")
            }
        };

        let checkpointing = if let Some(strategy) = checkpointing {
            Some(strategy)
        } else {
            Some(crate::CheckpointingStrategy::None)
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

// NOTE: placeholder for autodiff module requirements
#[cfg(not(feature = "autodiff"))]
impl AutodiffBackend for Dispatch {
    type InnerBackend = Dispatch;

    type Gradients = bool;

    fn backward(_tensor: DispatchTensor) -> Self::Gradients {
        unimplemented!("Requires `autodiff` feature")
    }

    fn grad(_tensor: &DispatchTensor, _grads: &Self::Gradients) -> Option<DispatchTensor> {
        unimplemented!("Requires `autodiff` feature")
    }

    fn grad_remove(
        _tensor: &DispatchTensor,
        _grads: &mut Self::Gradients,
    ) -> Option<DispatchTensor> {
        unimplemented!("Requires `autodiff` feature")
    }

    fn grad_replace(_tensor: &DispatchTensor, _grads: &mut Self::Gradients, _grad: DispatchTensor) {
        unimplemented!("Requires `autodiff` feature")
    }

    fn inner(_tensor: DispatchTensor) -> DispatchTensor {
        unimplemented!("Requires `autodiff` feature")
    }

    fn int_inner(_tensor: DispatchTensor) -> DispatchTensor {
        unimplemented!("Requires `autodiff` feature")
    }

    fn bool_inner(_tensor: DispatchTensor) -> DispatchTensor {
        unimplemented!("Requires `autodiff` feature")
    }

    fn q_inner(_tensor: DispatchTensor) -> DispatchTensor {
        unimplemented!("Requires `autodiff` feature")
    }

    fn from_inner(_tensor: DispatchTensor) -> DispatchTensor {
        unimplemented!("Requires `autodiff` feature")
    }

    fn int_from_inner(_tensor: DispatchTensor) -> DispatchTensor {
        unimplemented!("Requires `autodiff` feature")
    }

    fn bool_from_inner(_tensor: DispatchTensor) -> DispatchTensor {
        unimplemented!("Requires `autodiff` feature")
    }

    fn q_from_inner(_tensor: DispatchTensor) -> DispatchTensor {
        unimplemented!("Requires `autodiff` feature")
    }
}

impl DispatchTensorKind {
    pub(crate) fn device(&self) -> DispatchDevice {
        match self {
            #[cfg(feature = "cpu")]
            DispatchTensorKind::Cpu(tensor) => DispatchDevice::Cpu(tensor.device()),
            #[cfg(feature = "cuda")]
            DispatchTensorKind::Cuda(tensor) => DispatchDevice::Cuda(tensor.device()),
            #[cfg(feature = "metal")]
            DispatchTensorKind::Metal(tensor) => DispatchDevice::Metal(tensor.device()),
            #[cfg(feature = "rocm")]
            DispatchTensorKind::Rocm(tensor) => DispatchDevice::Rocm(tensor.device()),
            #[cfg(feature = "vulkan")]
            DispatchTensorKind::Vulkan(tensor) => DispatchDevice::Vulkan(tensor.device()),
            #[cfg(feature = "wgpu")]
            DispatchTensorKind::Wgpu(tensor) => DispatchDevice::Wgpu(tensor.device()),
            #[cfg(feature = "webgpu")]
            DispatchTensorKind::WebGpu(tensor) => DispatchDevice::WebGpu(tensor.device()),
            #[cfg(feature = "flex")]
            DispatchTensorKind::Flex(tensor) => DispatchDevice::Flex(tensor.device()),
            #[cfg(any(feature = "ndarray", default_backend))]
            DispatchTensorKind::NdArray(tensor) => DispatchDevice::NdArray(tensor.device()),
            #[cfg(feature = "tch")]
            DispatchTensorKind::LibTorch(tensor) => DispatchDevice::LibTorch(tensor.device()),
            #[cfg(feature = "remote")]
            DispatchTensorKind::Remote(tensor) => DispatchDevice::Remote(tensor.device()),
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(tensor) => DispatchDevice::autodiff(tensor.device()),
        }
    }
}

impl DispatchTensor {
    pub(crate) fn device(&self) -> DispatchDevice {
        #[allow(unused_mut)]
        let mut device = self.kind.device();

        // TODO: should int and bool kinds return an autodiff device?
        // It would be much easier once there is a single underlying primitive type, which
        // we can wrap with Autodiff in all cases.

        #[cfg(feature = "autodiff")]
        if let DispatchDevice::Autodiff(device) = &mut device
            && let Some(checkpointing) = &self.checkpointing
        {
            device.checkpointing = *checkpointing;
        }

        device
    }
}

impl Dispatch {
    /// List all available devices of the specified [type id](DispatchDeviceId).
    pub fn enumerate(type_id: DispatchDeviceId) -> Vec<DispatchDevice> {
        // TODO: right now this assumes `type_id = 0`, but WgpuDevice and LibTorchDevice have other types.
        match type_id {
            #[cfg(feature = "cpu")]
            DispatchDeviceId::Cpu => vec![CpuDevice.into()],
            #[cfg(feature = "cuda")]
            DispatchDeviceId::Cuda => (0..Cuda::device_count(0))
                .map(|i| CudaDevice::new(i).into())
                .collect(),
            #[cfg(feature = "metal")]
            DispatchDeviceId::Metal => (0..Metal::device_count(0))
                .map(|i| DispatchDevice::Metal(WgpuDevice::DiscreteGpu(i)))
                .collect(),
            #[cfg(feature = "rocm")]
            DispatchDeviceId::Rocm => (0..Rocm::device_count(0))
                .map(|i| RocmDevice::new(i).into())
                .collect(),
            #[cfg(feature = "vulkan")]
            DispatchDeviceId::Vulkan => (0..Vulkan::device_count(0))
                .map(|i| DispatchDevice::Vulkan(WgpuDevice::DiscreteGpu(i)))
                .collect(),
            #[cfg(feature = "wgpu")]
            DispatchDeviceId::Wgpu => (0..Wgpu::device_count(0))
                .map(|i| DispatchDevice::Wgpu(WgpuDevice::DiscreteGpu(i)))
                .collect(),
            #[cfg(feature = "webgpu")]
            DispatchDeviceId::WebGpu => (0..WebGpu::device_count(0))
                .map(|i| DispatchDevice::WebGpu(WgpuDevice::DiscreteGpu(i)))
                .collect(),
            #[cfg(feature = "flex")]
            DispatchDeviceId::Flex => vec![FlexDevice.into()],
            #[cfg(any(feature = "ndarray", default_backend))]
            DispatchDeviceId::NdArray => vec![NdArrayDevice::Cpu.into()],
            #[cfg(feature = "tch")]
            DispatchDeviceId::LibTorch => (0..LibTorch::device_count(0))
                .map(|i| LibTorchDevice::Cuda(i).into())
                .collect(),
            #[cfg(feature = "remote")]
            // Remote devices are user-supplied addresses, not enumerable hardware.
            DispatchDeviceId::Remote => Vec::new(),
            _ => unreachable!("No backend feature enabled."),
        }
    }
}
