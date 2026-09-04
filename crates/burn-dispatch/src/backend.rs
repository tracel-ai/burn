use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
#[cfg(cube_backend)]
use burn_backend::cubecl::{Device as CubeDevice, RuntimeId};

// The cubecl runtimes — `cpu` among them — enumerate through `cube_devices` rather than a
// `vec![]` literal, so only the backends that still list a fixed device need this.
#[cfg(any(feature = "ndarray", feature = "flex", default_backend))]
use alloc::vec;

#[cfg(feature = "autodiff")]
use burn_backend::distributed::{DistributedParamId, DistributedParams};
use burn_backend::{
    AutodiffBackend, Backend, BackendGraph, BackendTypes, DType, ExecutionError,
    InstallMemoryPoolsError, MemoryPoolLayout, MemoryPoolUsage, SlicedPoolReport,
};

/// A captured graph from one of the dispatched backends (see
/// [`BackendTypes::GraphPrimitive`]).
///
/// Like [`DispatchTensorKind`], one variant per enabled backend: the graph is
/// captured by, and can only replay on, the backend it was recorded on.
#[derive(Debug, Clone)]
pub enum DispatchGraph {
    /// A graph captured on the [cubecl backend](Cube).
    #[cfg(cube_backend)]
    Cube(BackendGraph<Cube>),

    /// A graph captured on the [Flex backend](Flex).
    #[cfg(any(feature = "flex", default_backend))]
    Flex(BackendGraph<Flex>),

    /// A graph captured on the [NdArray backend](NdArray).
    #[cfg(feature = "ndarray")]
    NdArray(BackendGraph<NdArray>),

    /// A graph captured on the [LibTorch backend](LibTorch).
    #[cfg(feature = "tch")]
    LibTorch(BackendGraph<LibTorch>),

    /// A graph captured on the [Remote backend](Remote).
    #[cfg(feature = "remote")]
    Remote(BackendGraph<Remote>),
    /// A graph captured by the non-executing capture backend.
    #[cfg(feature = "capture")]
    Capture(BackendGraph<Capture>),
}

/// The error returned when a graph operation cannot be dispatched.
fn graph_dispatch_err(reason: alloc::string::String) -> ExecutionError {
    ExecutionError::WithContext { reason }
}

/// Match arm generator for [`Backend::graph_stop_capture`] on [`Dispatch`]:
/// each backend's captured graph is wrapped in its [`DispatchGraph`] variant.
macro_rules! graph_stop_capture_arms {
    ($device:expr; $([$Backend:ident, $cfg:meta]),*) => {
        match $device {
            $(
                #[cfg($cfg)]
                $crate::DispatchDevice::$Backend(device) => {
                    <$crate::backends::$Backend as Backend>::graph_stop_capture(device)
                        .map(DispatchGraph::$Backend)
                }
            )*
            #[allow(unreachable_patterns)]
            other => Err(graph_dispatch_err(format!(
                "Graph capture is not supported for device {other:?}"
            ))),
        }
    };
}

/// Match arm generator for [`Backend::graph_replay`] on [`Dispatch`]: the graph
/// variant must match the device's backend, since a graph only replays on the
/// backend that captured it.
macro_rules! graph_replay_arms {
    ($device:expr, $graph:expr; $([$Backend:ident, $cfg:meta]),*) => {
        match ($device, $graph) {
            $(
                #[cfg($cfg)]
                ($crate::DispatchDevice::$Backend(device), DispatchGraph::$Backend(graph)) => {
                    // Safety: forwarded verbatim from `Dispatch::graph_replay`'s
                    // own contract.
                    unsafe {
                        <$crate::backends::$Backend as Backend>::graph_replay(device, graph)
                    }
                }
            )*
            #[allow(unreachable_patterns)]
            (device, _) => Err(graph_dispatch_err(format!(
                "The graph was not captured on the backend of device {device:?}"
            ))),
        }
    };
}

#[cfg(feature = "autodiff")]
macro_rules! is_tracked_arms {
    ($tensor:expr; $([$Backend:ident, $cfg:meta]),*) => {
        match &$tensor.kind {
            DispatchTensorKind::Autodiff(inner) => match &**inner {
                $(
                    #[cfg($cfg)]
                    DispatchTensorKind::$Backend(tensor) => tensor.as_autodiff().is_tracked(),
                )*
                DispatchTensorKind::Autodiff(_) => {
                    unreachable!("Autodiff should not wrap an autodiff tensor")
                }
                #[allow(unreachable_patterns)]
                _ => false,
            },
            _ => false,
        }
    };
}

#[cfg(feature = "autodiff")]
use alloc::boxed::Box;
#[cfg(feature = "autodiff")]
use burn_autodiff::grads::Gradients;

#[cfg(feature = "autodiff")]
use crate::DispatchAutodiffContext;
#[allow(unused)]
use crate::DispatchDeviceId;
#[allow(unused)]
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
/// let device = DispatchDevice::Cube(cubecl::Device::Cuda(Default::default()));
///
/// // Create a tensor using the global backend
/// let t = Tensor::<Dispatch, 2>::zeros([128, 128], &device);
/// ```
#[derive(Debug, Default, Clone)]
pub struct Dispatch;

#[cfg(feature = "autodiff")]
impl Dispatch {
    /// Returns whether an autodiff tensor participates in a recorded graph.
    #[doc(hidden)]
    pub fn is_tracked(tensor: &DispatchTensor) -> bool {
        backend_list!(is_tracked_arms, tensor)
    }
}

impl BackendTypes for Dispatch {
    type Device = DispatchDevice;

    type FloatTensorPrimitive = DispatchTensor;
    type IntTensorPrimitive = DispatchTensor;
    type BoolTensorPrimitive = DispatchTensor;
    type QuantizedTensorPrimitive = DispatchTensor;

    type GraphPrimitive = DispatchGraph;
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

    fn graph_prepare(device: &Self::Device) -> Result<(), ExecutionError> {
        dispatch_device!(device, |device| B::graph_prepare(device))
    }

    fn graph_start_capture(device: &Self::Device) -> Result<(), ExecutionError> {
        dispatch_device!(device, |device| B::graph_start_capture(device))
    }

    fn graph_stop_capture(device: &Self::Device) -> Result<DispatchGraph, ExecutionError> {
        backend_list!(graph_stop_capture_arms, device)
    }

    unsafe fn graph_replay(
        device: &Self::Device,
        graph: &DispatchGraph,
    ) -> Result<(), ExecutionError> {
        backend_list!(graph_replay_arms, device, graph)
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
            #[cfg(cube_backend)]
            DispatchDeviceId::Cube => Cube::device_count(backend_type_id),
            #[cfg(any(feature = "flex", default_backend))]
            DispatchDeviceId::Flex => Flex::device_count(backend_type_id),
            #[cfg(feature = "ndarray")]
            DispatchDeviceId::NdArray => NdArray::device_count(backend_type_id),
            #[cfg(feature = "tch")]
            DispatchDeviceId::LibTorch => LibTorch::device_count(backend_type_id),
            #[cfg(feature = "remote")]
            DispatchDeviceId::Remote => Remote::device_count(backend_type_id),
            #[cfg(feature = "capture")]
            DispatchDeviceId::Capture => Capture::device_count(backend_type_id),
            _ => unreachable!("No backend feature enabled."),
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

    fn memory_install_pools(
        device: &Self::Device,
        layout: MemoryPoolLayout,
    ) -> Result<(), InstallMemoryPoolsError> {
        dispatch_device!(device, |device| B::memory_install_pools(
            device,
            layout.clone()
        ))
    }

    fn memory_pool_report(device: &Self::Device) -> Option<Vec<SlicedPoolReport>> {
        dispatch_device!(device, |device| B::memory_pool_report(device))
    }

    fn memory_pool_usage(device: &Self::Device) -> Option<MemoryPoolUsage> {
        dispatch_device!(device, |device| B::memory_pool_usage(device))
    }

    fn staging<'a, Iter>(data: Iter, device: &Self::Device)
    where
        Iter: Iterator<Item = &'a mut burn_backend::TensorData>,
    {
        dispatch_device!(device, |device| B::staging(data, device))
    }

    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool {
        dispatch_device!(device, |device| B::supports_dtype(device, dtype))
    }

    fn flush(device: &Self::Device) {
        dispatch_device!(device, |device| B::flush(device))
    }
}

#[cfg(feature = "autodiff")]
fn disable_autodiff_context(context: DispatchAutodiffContext) -> DispatchAutodiffContext {
    assert!(
        matches!(context, DispatchAutodiffContext::Enabled(_)),
        "tensor is already on the inner backend"
    );
    DispatchAutodiffContext::Disabled
}

#[cfg(feature = "autodiff")]
fn enable_autodiff_context(context: DispatchAutodiffContext) -> DispatchAutodiffContext {
    assert_eq!(
        context,
        DispatchAutodiffContext::Disabled,
        "tensor is already associated with autodiff"
    );
    DispatchAutodiffContext::Enabled(crate::GradientCheckpointingStrategy::Disabled)
}

#[cfg(feature = "autodiff")]
impl AutodiffBackend for Dispatch {
    type InnerBackend = Dispatch;

    type Gradients = Gradients;

    fn backward(tensor: DispatchTensor) -> Self::Gradients {
        let DispatchTensor { kind, .. } = tensor;

        match kind {
            DispatchTensorKind::Autodiff(tensor) => match *tensor {
                #[cfg(cube_backend)]
                DispatchTensorKind::Cube(tensor) => tensor.autodiff().backward(),
                #[cfg(any(feature = "flex", default_backend))]
                DispatchTensorKind::Flex(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "ndarray")]
                DispatchTensorKind::NdArray(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "tch")]
                DispatchTensorKind::LibTorch(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "remote")]
                DispatchTensorKind::Remote(tensor) => tensor.autodiff().backward(),
                #[cfg(feature = "capture")]
                DispatchTensorKind::Capture(_) => {
                    panic!("Capture tensors do not support autodiff")
                }
                DispatchTensorKind::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn grad(tensor: &DispatchTensor, grads: &Self::Gradients) -> Option<DispatchTensor> {
        let DispatchTensor { kind, .. } = tensor;
        let grad: Option<DispatchTensorKind> = match &kind {
            DispatchTensorKind::Autodiff(inner_kind) => match &**inner_kind {
                #[cfg(cube_backend)]
                DispatchTensorKind::Cube(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Cube(crate::BackendTensor::Float(t))),
                #[cfg(any(feature = "flex", default_backend))]
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
                #[cfg(feature = "remote")]
                DispatchTensorKind::Remote(tensor) => tensor
                    .as_autodiff()
                    .grad(grads)
                    .map(|t| DispatchTensorKind::Remote(crate::BackendTensor::Float(t))),
                #[cfg(feature = "capture")]
                DispatchTensorKind::Capture(_) => {
                    panic!("Capture tensors do not support autodiff")
                }
                DispatchTensorKind::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        };
        grad.map(|kind| DispatchTensor {
            kind,
            autodiff: DispatchAutodiffContext::Disabled,
        })
    }

    fn grad_remove(tensor: &DispatchTensor, grads: &mut Self::Gradients) -> Option<DispatchTensor> {
        let DispatchTensor { kind, .. } = tensor;
        let grad: Option<DispatchTensorKind> = match &kind {
            DispatchTensorKind::Autodiff(inner_kind) => match &**inner_kind {
                #[cfg(cube_backend)]
                DispatchTensorKind::Cube(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Cube(crate::BackendTensor::Float(t))),
                #[cfg(any(feature = "flex", default_backend))]
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
                #[cfg(feature = "remote")]
                DispatchTensorKind::Remote(tensor) => tensor
                    .as_autodiff()
                    .grad_remove(grads)
                    .map(|t| DispatchTensorKind::Remote(crate::BackendTensor::Float(t))),
                #[cfg(feature = "capture")]
                DispatchTensorKind::Capture(_) => {
                    panic!("Capture tensors do not support autodiff")
                }
                DispatchTensorKind::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        };
        grad.map(|kind| DispatchTensor {
            kind,
            autodiff: DispatchAutodiffContext::Disabled,
        })
    }

    fn grad_replace(tensor: &DispatchTensor, grads: &mut Self::Gradients, grad: DispatchTensor) {
        let DispatchTensor { kind, .. } = tensor;
        let DispatchTensor {
            kind: grad,
            autodiff,
        } = grad;
        assert_eq!(
            autodiff,
            DispatchAutodiffContext::Disabled,
            "replacement gradients must use the inner backend"
        );

        match &kind {
            DispatchTensorKind::Autodiff(inner_kind) => match (&**inner_kind, grad) {
                #[cfg(cube_backend)]
                (DispatchTensorKind::Cube(tensor), DispatchTensorKind::Cube(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(any(feature = "flex", default_backend))]
                (DispatchTensorKind::Flex(tensor), DispatchTensorKind::Flex(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "ndarray")]
                (DispatchTensorKind::NdArray(tensor), DispatchTensorKind::NdArray(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                #[cfg(feature = "remote")]
                (DispatchTensorKind::Remote(tensor), DispatchTensorKind::Remote(grad)) => {
                    tensor.as_autodiff().grad_replace(grads, grad.float())
                }
                (DispatchTensorKind::Autodiff(_), _) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
                // TODO: distributed message?
                (t, g) => panic!(
                    "The provided tensors are not on the same backend. Got backends {t:?} and {g:?}."
                ),
            },
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn inner(tensor: DispatchTensor) -> DispatchTensor {
        let DispatchTensor { kind, autodiff } = tensor;
        assert!(
            matches!(autodiff, DispatchAutodiffContext::Enabled(_)),
            "Requires autodiff tensor."
        );

        let kind = match kind {
            DispatchTensorKind::Autodiff(inner_kind) => match *inner_kind {
                #[cfg(cube_backend)]
                DispatchTensorKind::Cube(tensor) => DispatchTensorKind::Cube(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(any(feature = "flex", default_backend))]
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
                #[cfg(feature = "remote")]
                DispatchTensorKind::Remote(tensor) => DispatchTensorKind::Remote(
                    crate::BackendTensor::Float(tensor.autodiff().primitive),
                ),
                #[cfg(feature = "capture")]
                DispatchTensorKind::Capture(_) => {
                    panic!("Capture tensors do not support autodiff")
                }
                DispatchTensorKind::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        };
        DispatchTensor {
            kind,
            autodiff: DispatchAutodiffContext::Disabled,
        }
    }

    fn int_inner(mut tensor: DispatchTensor) -> DispatchTensor {
        tensor.autodiff = disable_autodiff_context(tensor.autodiff);
        tensor
    }

    fn bool_inner(mut tensor: DispatchTensor) -> DispatchTensor {
        tensor.autodiff = disable_autodiff_context(tensor.autodiff);
        tensor
    }

    fn q_inner(mut tensor: DispatchTensor) -> DispatchTensor {
        tensor.autodiff = disable_autodiff_context(tensor.autodiff);
        tensor
    }

    fn from_inner(tensor: DispatchTensor) -> DispatchTensor {
        let DispatchTensor { kind, autodiff } = tensor;
        assert_eq!(
            autodiff,
            DispatchAutodiffContext::Disabled,
            "tensor is already associated with autodiff"
        );

        let kind = match kind {
            #[cfg(cube_backend)]
            DispatchTensorKind::Cube(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Cube(
                    crate::BackendTensor::Autodiff(Autodiff::<Cube>::from_inner(tensor.float())),
                )))
            }
            #[cfg(any(feature = "flex", default_backend))]
            DispatchTensorKind::Flex(tensor) => {
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Flex(
                    crate::BackendTensor::Autodiff(Autodiff::<Flex>::from_inner(tensor.float())),
                )))
            }
            #[cfg(feature = "ndarray")]
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
            #[cfg(feature = "capture")]
            DispatchTensorKind::Capture(_) => {
                panic!("Capture tensors do not support autodiff")
            }
            DispatchTensorKind::Autodiff(_) => {
                panic!("Autodiff should not wrap an autodiff tensor.")
            }
        };

        DispatchTensor {
            kind,
            autodiff: DispatchAutodiffContext::Enabled(
                crate::GradientCheckpointingStrategy::Disabled,
            ),
        }
    }

    fn int_from_inner(mut tensor: DispatchTensor) -> DispatchTensor {
        tensor.autodiff = enable_autodiff_context(tensor.autodiff);
        tensor
    }

    fn bool_from_inner(mut tensor: DispatchTensor) -> DispatchTensor {
        tensor.autodiff = enable_autodiff_context(tensor.autodiff);
        tensor
    }

    fn q_from_inner(mut tensor: DispatchTensor) -> DispatchTensor {
        tensor.autodiff = enable_autodiff_context(tensor.autodiff);
        tensor
    }

    // Only the collective-capable backends (Cube/Remote) carry distributed params; in builds
    // without them the match arms cfg out, leaving the bindings unused and the tail unreachable.
    #[allow(unused_variables, unreachable_code)]
    fn set_distributed_params(
        tensor: DispatchTensor,
        param_id: DistributedParamId,
    ) -> DispatchTensor {
        let DispatchTensor { kind, autodiff } = tensor;
        assert!(
            matches!(autodiff, DispatchAutodiffContext::Enabled(_)),
            "Requires autodiff tensor."
        );

        let kind = match kind {
            DispatchTensorKind::Autodiff(inner_kind) => match *inner_kind {
                #[cfg(cube_backend)]
                DispatchTensorKind::Cube(tensor) => {
                    DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Cube(
                        crate::BackendTensor::Autodiff(Autodiff::<Cube>::set_distributed_params(
                            tensor.as_autodiff().clone(),
                            param_id,
                        )),
                    )))
                }
                #[cfg(feature = "remote")]
                DispatchTensorKind::Remote(tensor) => {
                    DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Remote(
                        crate::BackendTensor::Autodiff(Autodiff::<Remote>::set_distributed_params(
                            tensor.as_autodiff().clone(),
                            param_id,
                        )),
                    )))
                }
                DispatchTensorKind::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
                #[allow(unreachable_patterns)]
                other => {
                    panic!("Distributed operations are not supported for tensor kind {other:?}")
                }
            },
            _ => panic!("Requires autodiff tensor."),
        };

        DispatchTensor { kind, autodiff }
    }

    #[allow(unused_variables)]
    fn distributed_params(tensor: &DispatchTensor) -> Option<DistributedParams> {
        let DispatchTensor { kind, autodiff: _ } = tensor;

        match &kind {
            DispatchTensorKind::Autodiff(inner_kind) => match &**inner_kind {
                #[cfg(cube_backend)]
                DispatchTensorKind::Cube(tensor) => {
                    tensor.as_autodiff().node.distributed_params.clone()
                }
                #[cfg(feature = "remote")]
                DispatchTensorKind::Remote(tensor) => {
                    tensor.as_autodiff().node.distributed_params.clone()
                }

                DispatchTensorKind::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
                // Backends without distributed support never carry distributed params.
                #[allow(unreachable_patterns)]
                _ => None,
            },
            _ => panic!("Requires autodiff tensor."),
        }
    }

    #[allow(unused_variables)]
    fn is_distributed(tensor: &DispatchTensor) -> bool {
        let DispatchTensor { kind, autodiff: _ } = tensor;

        match &kind {
            DispatchTensorKind::Autodiff(inner_kind) => match &**inner_kind {
                #[cfg(cube_backend)]
                DispatchTensorKind::Cube(tensor) => {
                    tensor.as_autodiff().node.distributed_params.is_some()
                }
                #[cfg(feature = "remote")]
                DispatchTensorKind::Remote(tensor) => {
                    tensor.as_autodiff().node.distributed_params.is_some()
                }

                DispatchTensorKind::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff tensor.")
                }
                // Backends without distributed support are never distributed.
                #[allow(unreachable_patterns)]
                _ => false,
            },
            _ => panic!("Requires autodiff tensor."),
        }
    }
}

#[cfg(all(test, feature = "autodiff", any(feature = "flex", default_backend)))]
mod autodiff_context_tests {
    use super::*;
    use crate::{DispatchAutodiffContext, DispatchTensorKind, GradientCheckpointingStrategy};
    use alloc::vec;
    use burn_backend::{
        TensorData, TensorMetadata, TensorPrimitive,
        ops::{BoolTensorOps, FloatTensorOps, IntTensorOps, ModuleOps, QTensorOps},
        quantization::QuantScheme,
        tensor::{FloatTensor, IntTensor},
    };
    use burn_backend_extension::backend_dispatch;

    #[backend_dispatch]
    impl Dispatch {
        fn direct_float(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
            tensor
        }

        fn conditional_float_route(
            int: IntTensor<Self>,
            float: Option<FloatTensor<Self>>,
        ) -> IntTensor<Self> {
            // `optional_float_only_uses_autodiff_when_present` exercises both routes: an enabled
            // int alone uses the concrete backend, while a present float selects `Autodiff<B>`.
            assert_eq!(float.is_some(), B::ad_enabled(&int.device()));
            int
        }

        fn optional_routing_tensor(
            first: Option<FloatTensor<Self>>,
            second: Option<FloatTensor<Self>>,
        ) -> FloatTensor<Self> {
            first.or(second).expect("test requires one tensor")
        }

        fn vector_routing_tensor(
            first: Vec<IntTensor<Self>>,
            second: Vec<IntTensor<Self>>,
        ) -> IntTensor<Self> {
            first
                .into_iter()
                .chain(second)
                .next()
                .expect("test requires one tensor")
        }
    }

    fn device(strategy: GradientCheckpointingStrategy) -> DispatchDevice {
        DispatchDevice::autodiff_with_gradient_checkpointing(
            DispatchDevice::Flex(Default::default()),
            strategy,
        )
    }

    fn inner_device() -> DispatchDevice {
        DispatchDevice::Flex(Default::default())
    }

    fn float(values: [f32; 2], device: &DispatchDevice) -> DispatchTensor {
        Dispatch::float_from_data(TensorData::from(values), device)
    }

    fn float_2d(values: [[f32; 2]; 2], device: &DispatchDevice) -> DispatchTensor {
        Dispatch::float_from_data(TensorData::from(values), device)
    }

    fn assert_enabled_float(tensor: &DispatchTensor, strategy: GradientCheckpointingStrategy) {
        assert_eq!(tensor.autodiff, DispatchAutodiffContext::Enabled(strategy));
        assert!(matches!(tensor.kind, DispatchTensorKind::Autodiff(_)));
        let DispatchDevice::Autodiff(device) = tensor.device() else {
            panic!("enabled float should report an autodiff device")
        };
        assert_eq!(device.checkpointing, strategy);
    }

    #[test]
    fn enabled_float_creation_is_untracked_for_both_strategies() {
        for strategy in [
            GradientCheckpointingStrategy::Disabled,
            GradientCheckpointingStrategy::Balanced,
        ] {
            let tensor = float([1.0, 2.0], &device(strategy));
            assert_enabled_float(&tensor, strategy);
            assert!(!Dispatch::float_is_require_grad(&tensor));
        }
    }

    #[test]
    fn enabled_int_bool_and_float_conversions_preserve_association() {
        let strategy = GradientCheckpointingStrategy::Balanced;
        let device = device(strategy);
        let int = Dispatch::int_from_data(TensorData::from([1i32, 2]), &device);
        assert_eq!(int.autodiff, DispatchAutodiffContext::Enabled(strategy));
        assert!(!matches!(int.kind, DispatchTensorKind::Autodiff(_)));
        assert!(matches!(int.device(), DispatchDevice::Autodiff(_)));

        let converted = Dispatch::int_into_float(int, burn_backend::FloatDType::F32);
        assert_enabled_float(&converted, strategy);

        let boolean = Dispatch::bool_from_data(TensorData::from([true, false]), &device);
        assert_eq!(boolean.autodiff, DispatchAutodiffContext::Enabled(strategy));
        assert!(!matches!(boolean.kind, DispatchTensorKind::Autodiff(_)));
        let converted = Dispatch::bool_into_float(boolean, burn_backend::FloatDType::F32);
        assert_enabled_float(&converted, strategy);
    }

    #[test]
    fn uniform_contexts_propagate_through_fixed_arity_and_vector_ops() {
        let strategy = GradientCheckpointingStrategy::Disabled;
        let lhs = float([1.0, 2.0], &device(strategy));
        let rhs = float([3.0, 4.0], &device(strategy));

        let added = Dispatch::float_add(lhs.clone(), rhs.clone());
        assert_enabled_float(&added, strategy);

        let concatenated = Dispatch::float_cat(vec![lhs.clone(), rhs], 0);
        assert_enabled_float(&concatenated, strategy);

        let normalized = Dispatch::layer_norm(
            lhs,
            float([1.0, 1.0], &device(strategy)),
            Some(float([0.0, 0.0], &device(strategy))),
            1e-5,
        );
        assert_enabled_float(&normalized, strategy);
    }

    #[test]
    fn optional_and_vector_routing_candidates_skip_empty_inputs() {
        let strategy = GradientCheckpointingStrategy::Balanced;
        let float = float([1.0, 2.0], &device(strategy));
        let output = Dispatch::optional_routing_tensor(None, Some(float));
        assert_enabled_float(&output, strategy);

        let int = Dispatch::int_from_data(TensorData::from([1i32, 2]), &device(strategy));
        let output = Dispatch::vector_routing_tensor(vec![], vec![int]);
        assert_eq!(output.autodiff, DispatchAutodiffContext::Enabled(strategy));
    }

    #[test]
    fn optional_float_only_uses_autodiff_when_present() {
        let strategy = GradientCheckpointingStrategy::Balanced;
        let int = Dispatch::int_from_data(TensorData::from([1i32, 2]), &device(strategy));
        let output = Dispatch::conditional_float_route(int, None);
        assert_eq!(output.autodiff, DispatchAutodiffContext::Enabled(strategy));

        let int = Dispatch::int_from_data(TensorData::from([1i32, 2]), &inner_device());
        let float = float([1.0, 2.0], &device(strategy));
        let output = Dispatch::conditional_float_route(int, Some(float));
        assert_eq!(output.autodiff, DispatchAutodiffContext::Enabled(strategy));
    }

    #[test]
    #[should_panic(expected = "an enabled float tensor must use an autodiff primitive")]
    fn direct_routing_tensor_validates_its_context() {
        let mut malformed = float([1.0, 2.0], &inner_device());
        malformed.autodiff =
            DispatchAutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled);
        let _ = Dispatch::direct_float(malformed);
    }

    #[test]
    #[should_panic(expected = "Gradient checkpointing strategy mismatch")]
    fn fixed_arity_inputs_reject_mismatched_checkpointing_strategies() {
        let lhs = float([1.0, 2.0], &device(GradientCheckpointingStrategy::Balanced));
        let rhs = float([3.0, 4.0], &device(GradientCheckpointingStrategy::Disabled));
        let _ = Dispatch::float_add(lhs, rhs);
    }

    #[test]
    fn disabled_and_enabled_integer_contexts_merge_in_both_orders() {
        for strategy in [
            GradientCheckpointingStrategy::Disabled,
            GradientCheckpointingStrategy::Balanced,
        ] {
            let lhs = Dispatch::int_from_data(TensorData::from([1i32, 2]), &inner_device());
            let rhs = Dispatch::int_from_data(TensorData::from([3i32, 4]), &device(strategy));
            let output = Dispatch::int_add(lhs, rhs);
            assert_eq!(output.autodiff, DispatchAutodiffContext::Enabled(strategy));

            let lhs = Dispatch::int_from_data(TensorData::from([1i32, 2]), &device(strategy));
            let rhs = Dispatch::int_from_data(TensorData::from([3i32, 4]), &inner_device());
            let output = Dispatch::int_add(lhs, rhs);
            assert_eq!(output.autodiff, DispatchAutodiffContext::Enabled(strategy));
        }
    }

    #[test]
    fn disabled_and_enabled_float_contexts_merge_in_both_orders() {
        for strategy in [
            GradientCheckpointingStrategy::Disabled,
            GradientCheckpointingStrategy::Balanced,
        ] {
            let enabled = float([1.0, 2.0], &device(strategy));
            let disabled = float([3.0, 4.0], &inner_device());

            let output = Dispatch::float_add(enabled, disabled);
            assert_enabled_float(&output, strategy);

            let enabled = float([1.0, 2.0], &device(strategy));
            let disabled = float([3.0, 4.0], &inner_device());
            let output = Dispatch::float_add(disabled, enabled);
            assert_enabled_float(&output, strategy);
        }
    }

    #[test]
    fn disabled_and_enabled_contexts_merge_across_vector_inputs() {
        let strategy = GradientCheckpointingStrategy::Balanced;
        let disabled = float([1.0, 2.0], &inner_device());
        let enabled = float([3.0, 4.0], &device(strategy));

        let output = Dispatch::float_cat(vec![disabled, enabled], 0);
        assert_enabled_float(&output, strategy);
    }

    #[test]
    #[should_panic(expected = "Gradient checkpointing strategy mismatch")]
    fn vector_inputs_reject_mismatched_checkpointing_strategies() {
        let balanced = float([1.0, 2.0], &device(GradientCheckpointingStrategy::Balanced));
        let disabled = float([3.0, 4.0], &device(GradientCheckpointingStrategy::Disabled));
        let _ = Dispatch::float_cat(vec![balanced, disabled], 0);
    }

    #[test]
    #[should_panic(expected = "Gradient checkpointing strategy mismatch")]
    fn q_matmul_rejects_mismatched_checkpointing_strategies() {
        let balanced = Dispatch::quantize_dynamic(
            float([1.0, 2.0], &device(GradientCheckpointingStrategy::Balanced)),
            &QuantScheme::default(),
        );
        let disabled = Dispatch::quantize_dynamic(
            float([3.0, 4.0], &device(GradientCheckpointingStrategy::Disabled)),
            &QuantScheme::default(),
        );
        let _ = Dispatch::q_matmul(
            TensorPrimitive::QFloat(balanced),
            TensorPrimitive::QFloat(disabled),
        );
    }

    #[test]
    fn q_matmul_merges_disabled_and_enabled_contexts() {
        let strategy = GradientCheckpointingStrategy::Balanced;
        let disabled = Dispatch::quantize_dynamic(
            float_2d([[1.0, 2.0], [3.0, 4.0]], &inner_device()),
            &QuantScheme::default(),
        );
        let enabled = Dispatch::quantize_dynamic(
            float_2d([[1.0, 0.0], [0.0, 1.0]], &device(strategy)),
            &QuantScheme::default(),
        );
        let output = Dispatch::q_matmul(
            TensorPrimitive::QFloat(disabled),
            TensorPrimitive::QFloat(enabled),
        );
        let output = match output {
            TensorPrimitive::QFloat(output) | TensorPrimitive::Float(output) => output,
        };

        assert_eq!(output.autodiff, DispatchAutodiffContext::Enabled(strategy));
    }

    #[test]
    fn inner_transitions_clear_and_restore_context() {
        let tensor = float([1.0, 2.0], &device(GradientCheckpointingStrategy::Balanced));
        let inner = <Dispatch as AutodiffBackend>::inner(tensor);
        assert_eq!(inner.autodiff, DispatchAutodiffContext::Disabled);
        assert!(!matches!(inner.kind, DispatchTensorKind::Autodiff(_)));

        let enabled = <Dispatch as AutodiffBackend>::from_inner(inner);
        assert_enabled_float(&enabled, GradientCheckpointingStrategy::Disabled);

        let int = Dispatch::int_from_data(
            TensorData::from([1i32, 2]),
            &device(GradientCheckpointingStrategy::Balanced),
        );
        let int = <Dispatch as AutodiffBackend>::int_inner(int);
        assert_eq!(int.autodiff, DispatchAutodiffContext::Disabled);
        let int = <Dispatch as AutodiffBackend>::int_from_inner(int);
        assert_eq!(
            int.autodiff,
            DispatchAutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled)
        );
    }

    #[test]
    fn gradients_are_inner_backend_tensors() {
        let x = Dispatch::float_set_require_grad(
            float([2.0, 3.0], &device(GradientCheckpointingStrategy::Balanced)),
            true,
        );
        let output = Dispatch::float_mul(x.clone(), x.clone());
        let gradients = <Dispatch as AutodiffBackend>::backward(output);
        let gradient = <Dispatch as AutodiffBackend>::grad(&x, &gradients).unwrap();
        assert_eq!(gradient.autodiff, DispatchAutodiffContext::Disabled);
        assert!(!matches!(gradient.kind, DispatchTensorKind::Autodiff(_)));
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

impl Dispatch {
    /// List all available devices of the specified [type id](DispatchDeviceId).
    pub fn enumerate(type_id: DispatchDeviceId) -> Vec<DispatchDevice> {
        // TODO: right now this assumes `type_id = 0`, but WgpuDevice and LibTorchDevice have other types.
        match type_id {
            #[cfg(cube_backend)]
            DispatchDeviceId::Cube => CubeDevice::enumerate_all()
                .into_iter()
                .filter(|device| cube_runtime_enabled(device.runtime()))
                .map(DispatchDevice::Cube)
                .collect(),
            #[cfg(any(feature = "flex", default_backend))]
            DispatchDeviceId::Flex => vec![FlexDevice.into()],
            #[cfg(feature = "ndarray")]
            DispatchDeviceId::NdArray => vec![NdArrayDevice::Cpu.into()],
            #[cfg(feature = "tch")]
            DispatchDeviceId::LibTorch => (0..LibTorch::device_count(0))
                .map(|i| LibTorchDevice::Cuda(i).into())
                .collect(),
            #[cfg(feature = "remote")]
            // Remote devices are keyed by a network address, which the type-id-only
            // `enumerate` can't carry. Use [`Dispatch::enumerate_remote_websocket`] to list the devices
            // behind a given address.
            DispatchDeviceId::Remote => Vec::new(),
            #[cfg(feature = "capture")]
            // Capture devices are created together with a lifecycle handle and therefore
            // cannot be reconstructed from a type ID alone.
            DispatchDeviceId::Capture => Vec::new(),
            _ => unreachable!("No backend feature enabled."),
        }
    }

    /// List every device of the cubecl `runtime`.
    ///
    /// The cubecl runtimes share one backend, so [`enumerate`](Self::enumerate) with
    /// [`DispatchDeviceId::Cube`] lists all of them at once. A caller that asked for one
    /// runtime — CUDA, say — wants that runtime's devices and not the wgpu and CPU ones found
    /// beside them, which is what this narrows to. Empty for a runtime this build's features
    /// did not enable.
    #[cfg(cube_backend)]
    pub fn enumerate_cube(runtime: RuntimeId) -> Vec<DispatchDevice> {
        cube_devices(runtime)
            .into_iter()
            .map(DispatchDevice::Cube)
            .collect()
    }

    /// List every device hosted by the remote server at `address`.
    ///
    /// Unlike [`enumerate`](Self::enumerate), remote devices are identified by a network
    /// address rather than enumerable local hardware, so they need a dedicated entry point.
    /// Connecting to the server (required to learn its device count) happens here; see
    /// [`RemoteDevice::enumerate_websocket`].
    ///
    /// Websocket-only: Iroh peers are addressed by endpoint identity, not a URL string.
    #[cfg(feature = "remote-websocket")]
    pub fn enumerate_remote_websocket(address: &str) -> Vec<DispatchDevice> {
        RemoteDevice::enumerate_websocket(address)
            .into_iter()
            .map(DispatchDevice::Remote)
            .collect()
    }
}

/// Whether `runtime` is one this crate's features asked for.
///
/// `CubeDevice::enumerate_all` answers for the runtimes *cubecl* compiled in, and cargo unifies
/// features across a build: a workspace that also builds `burn-cuda` gives cubecl the CUDA
/// runtime even where this crate was built with only `wgpu`. What this crate hands out has to
/// follow its own features, so enumeration is filtered through here.
#[cfg(cube_backend)]
fn cube_runtime_enabled(runtime: RuntimeId) -> bool {
    match runtime {
        RuntimeId::Cuda => cfg!(feature = "cuda"),
        RuntimeId::Hip => cfg!(feature = "rocm"),
        RuntimeId::Wgpu => cfg!(any(
            feature = "wgpu",
            feature = "vulkan",
            feature = "metal",
            feature = "webgpu"
        )),
        RuntimeId::Cpu => cfg!(feature = "cpu"),
        // burn's `metal` feature is wgpu compiling to MSL; cubecl's native Metal runtime has no
        // burn feature to enable it.
        RuntimeId::Metal => false,
    }
}

/// The cubecl devices belonging to `runtime`, in enumeration order. Empty for a runtime this
/// build's features did not ask for, even where cubecl compiled it in.
#[cfg(cube_backend)]
pub(crate) fn cube_devices(runtime: RuntimeId) -> Vec<CubeDevice> {
    if !cube_runtime_enabled(runtime) {
        return Vec::new();
    }

    CubeDevice::enumerate_all()
        .into_iter()
        .filter(|device| device.runtime() == runtime)
        .collect()
}
