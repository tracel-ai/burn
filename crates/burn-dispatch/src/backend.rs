use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use burn_backend::RandomKey;
#[cfg(cube_backend)]
use burn_backend::cubecl::{Device as CubeDevice, RuntimeId};

// The cubecl runtimes — `cpu` among them — enumerate through `cube_devices` rather than a
// `vec![]` literal, so only the backends that still list a fixed device need this.
#[cfg(feature = "flex")]
use alloc::vec;

#[cfg(feature = "autodiff")]
use burn_backend::distributed::{DistributedParamId, DistributedParams};
use burn_backend::{
    AutodiffBackend, Backend, BackendGraph, BackendTypes, DType, DeviceDescription,
    DeviceTelemetry, EnvironmentBundle, EnvironmentError, EnvironmentId, EnvironmentOrigin,
    EnvironmentSummary, ExecutionError, ExecutionMode, ExecutionModeToken, ExecutionStatistics,
    HostTransfer, MemoryReport, Peak, PeakError, PeakKey, ProfileDuration, ProfileOptions,
    ProfileToken, PruneOptions, Pruned, SaveOptions,
};

/// A captured graph from one of the dispatched backends (see
/// [`BackendTypes::GraphPrimitive`]).
///
/// Like [`DispatchTensorKind`], one variant per enabled backend: the graph is
/// captured by, and can only replay on, the backend it was recorded on.
#[derive(Debug, Clone)]
pub enum DispatchGraph {
    #[cfg(not(backend_enabled))]
    #[doc(hidden)]
    Unavailable(crate::NoBackend),
    /// A graph captured on the [cubecl backend](Cube).
    #[cfg(cube_backend)]
    Cube(BackendGraph<Cube>),

    /// A graph captured on the [Flex backend](Flex).
    #[cfg(feature = "flex")]
    Flex(BackendGraph<Flex>),

    /// A graph captured on the [Remote backend](Remote).
    #[cfg(feature = "remote")]
    Remote(BackendGraph<Remote>),
    /// A graph captured on the [sharded compute device backend](ShardedCompute).
    #[cfg(feature = "shard")]
    ShardedCompute(BackendGraph<ShardedCompute>),
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
            DispatchTensorKind::Autodiff(tensor) => tensor.is_tracked(),
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
#[cfg(feature = "flex")]
use crate::devices::FlexDevice;
use crate::{DispatchDevice, DispatchTensor, backends::*};

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
    /// Returns whether an autodiff tensor's node is marked for graph participation.
    #[doc(hidden)]
    pub fn is_tracked(tensor: &DispatchTensor) -> bool {
        backend_list!(is_tracked_arms, tensor)
    }
}

/// The name of the backend behind `device`, one `autodiff<...>` per level.
fn dispatch_name(device: &DispatchDevice) -> String {
    #[cfg(feature = "autodiff")]
    if let DispatchDevice::Autodiff(device) = device {
        return format!("autodiff<{}>", dispatch_name(&device.inner));
    }
    dispatch_device!(device, |device| B::name(device))
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
        format!("dispatch<{}>", dispatch_name(device))
    }

    fn seed(device: &Self::Device, seed: u64) {
        dispatch_device!(device, |device| B::seed(device, seed))
    }

    fn next_random_key(device: &Self::Device) -> RandomKey {
        dispatch_device!(device, |device| B::next_random_key(device))
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        dispatch_device!(device, |device| B::sync(device))
    }

    fn profile<O: Send + 'static>(
        device: &Self::Device,
        options: ProfileOptions,
        func: impl FnOnce() -> O + Send,
    ) -> Result<(O, ProfileDuration), ExecutionError> {
        dispatch_device!(device, |device| B::profile(device, options, func))
    }

    fn profile_start(device: &Self::Device) -> Result<Option<ProfileToken>, ExecutionError> {
        dispatch_device!(device, |device| B::profile_start(device))
    }

    fn profile_end(
        device: &Self::Device,
        token: ProfileToken,
        options: ProfileOptions,
    ) -> Result<ProfileDuration, ExecutionError> {
        dispatch_device!(device, |device| B::profile_end(device, token, options))
    }

    fn profile_abandon(device: &Self::Device, token: ProfileToken) {
        dispatch_device!(device, |device| B::profile_abandon(device, token))
    }

    fn execution_mode_open(
        device: &Self::Device,
        mode: ExecutionMode,
    ) -> Result<Option<ExecutionModeToken>, ExecutionError> {
        dispatch_device!(device, |device| B::execution_mode_open(device, mode))
    }

    fn execution_mode_statistics(
        device: &Self::Device,
        token: ExecutionModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        dispatch_device!(device, |device| B::execution_mode_statistics(device, token))
    }

    fn execution_mode_close(
        device: &Self::Device,
        token: ExecutionModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        dispatch_device!(device, |device| B::execution_mode_close(device, token))
    }

    fn environment_create(
        device: &Self::Device,
        name: String,
        origin: EnvironmentOrigin,
    ) -> Result<EnvironmentId, EnvironmentError> {
        dispatch_device!(device, |device| B::environment_create(device, name, origin))
    }

    fn environment_activate(
        device: &Self::Device,
        id: EnvironmentId,
    ) -> Result<EnvironmentId, EnvironmentError> {
        dispatch_device!(device, |device| B::environment_activate(device, id))
    }

    fn environment_active(device: &Self::Device) -> Result<EnvironmentId, EnvironmentError> {
        dispatch_device!(device, |device| B::environment_active(device))
    }

    fn environment_save(
        device: &Self::Device,
        id: EnvironmentId,
        options: SaveOptions,
    ) -> Result<EnvironmentBundle, EnvironmentError> {
        dispatch_device!(device, |device| B::environment_save(device, id, options))
    }

    fn environment_summary(
        device: &Self::Device,
        id: EnvironmentId,
    ) -> Result<EnvironmentSummary, EnvironmentError> {
        dispatch_device!(device, |device| B::environment_summary(device, id))
    }

    fn environment_merge(
        device: &Self::Device,
        id: EnvironmentId,
        origin: EnvironmentOrigin,
    ) -> Result<(), EnvironmentError> {
        dispatch_device!(device, |device| B::environment_merge(device, id, origin))
    }

    fn environment_prune(
        device: &Self::Device,
        id: EnvironmentId,
        options: PruneOptions,
    ) -> Result<Pruned, EnvironmentError> {
        dispatch_device!(device, |device| B::environment_prune(device, id, options))
    }

    fn graph_prepare(device: &Self::Device) -> Result<(), ExecutionError> {
        dispatch_device!(device, |device| B::graph_prepare(device))
    }

    fn graph_end_run(device: &Self::Device) -> Result<bool, ExecutionError> {
        dispatch_device!(device, |device| B::graph_end_run(device))
    }

    fn graph_start_run(device: &Self::Device) -> Result<(), ExecutionError> {
        dispatch_device!(device, |device| B::graph_start_run(device))
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
        #[allow(unreachable_patterns)]
        match dispatch_id {
            #[cfg(cube_backend)]
            DispatchDeviceId::Cube => Cube::device_count(backend_type_id),
            #[cfg(feature = "flex")]
            DispatchDeviceId::Flex => Flex::device_count(backend_type_id),
            #[cfg(feature = "remote")]
            DispatchDeviceId::Remote => Remote::device_count(backend_type_id),
            #[cfg(feature = "shard")]
            DispatchDeviceId::ShardedCompute => ShardedCompute::device_count(backend_type_id),
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

    fn device_description(device: &Self::Device) -> Option<DeviceDescription> {
        dispatch_device!(device, |device| B::device_description(device))
    }

    fn measure_telemetry(device: &Self::Device) -> DeviceTelemetry {
        dispatch_device!(device, |device| B::measure_telemetry(device))
    }

    fn measure_memory(device: &Self::Device) -> Option<MemoryReport> {
        dispatch_device!(device, |device| B::measure_memory(device))
    }

    fn measure_peaks(device: &Self::Device, keys: &[PeakKey]) -> Vec<Result<Peak, PeakError>> {
        dispatch_device!(device, |device| B::measure_peaks(device, keys))
    }

    /// Unsupported between devices of two backends: no backend moves data across that link.
    fn measure_link(
        from: &Self::Device,
        to: &Self::Device,
        sizes: &[u64],
    ) -> Vec<Result<Peak, PeakError>> {
        dispatch_device_pair!(
            measured(from),
            measured(to),
            |from, to| B::measure_link(from, to, sizes),
            sizes.iter().map(|_| Err(PeakError::Unsupported)).collect()
        )
    }

    fn measure_host_link(
        device: &Self::Device,
        transfer: HostTransfer,
        sizes: &[u64],
    ) -> Vec<Result<Peak, PeakError>> {
        dispatch_device!(device, |device| B::measure_host_link(
            device, transfer, sizes
        ))
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

    fn flush(device: &Self::Device) -> Result<(), ExecutionError> {
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
/// Panics unless the tensor's concrete backend supports distributed gradients.
fn require_distributed_kind(kind: &DispatchTensorKind) {
    match kind {
        DispatchTensorKind::Autodiff(tensor) => require_distributed_kind(&tensor.primitive().kind),
        #[cfg(cube_backend)]
        DispatchTensorKind::Cube(_) => {}
        #[cfg(feature = "remote")]
        DispatchTensorKind::Remote(_) => {}
        #[allow(unreachable_patterns)]
        other => panic!("Distributed operations are not supported for tensor kind {other:?}"),
    }
}

#[cfg(feature = "autodiff")]
// Capture does not support autodiff. Without an execution backend, the dispatch
// arms below only panic, leaving gradient arguments unused and return code unreachable.
#[cfg_attr(not(executing_backend), allow(unused_variables, unreachable_code))]
impl AutodiffBackend for Dispatch {
    type InnerBackend = Dispatch;

    type Gradients = Gradients;

    fn backward(tensor: DispatchTensor) -> Self::Gradients {
        match tensor.kind {
            DispatchTensorKind::Autodiff(tensor) => tensor.backward(),
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn grad(tensor: &DispatchTensor, grads: &Self::Gradients) -> Option<DispatchTensor> {
        match &tensor.kind {
            DispatchTensorKind::Autodiff(tensor) => tensor.grad(grads),
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn grad_remove(tensor: &DispatchTensor, grads: &mut Self::Gradients) -> Option<DispatchTensor> {
        match &tensor.kind {
            DispatchTensorKind::Autodiff(tensor) => tensor.grad_remove(grads),
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn grad_replace(tensor: &DispatchTensor, grads: &mut Self::Gradients, grad: DispatchTensor) {
        match &tensor.kind {
            DispatchTensorKind::Autodiff(tensor) => tensor.grad_replace(grads, grad),
            _ => panic!("Requires autodiff tensor."),
        }
    }

    fn inner(tensor: DispatchTensor) -> DispatchTensor {
        assert!(
            matches!(tensor.autodiff, DispatchAutodiffContext::Enabled(_)),
            "Requires autodiff tensor."
        );

        match tensor.kind {
            DispatchTensorKind::Autodiff(tensor) => tensor.into_primitive(),
            _ => panic!("Requires autodiff tensor."),
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
        DispatchTensor {
            kind: DispatchTensorKind::Autodiff(Box::new(<crate::backends::Autodiff<
                crate::Dispatch,
            > as AutodiffBackend>::from_inner(
                tensor
            ))),
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

    fn set_distributed_params(
        tensor: DispatchTensor,
        param_id: DistributedParamId,
    ) -> DispatchTensor {
        let DispatchTensor { kind, autodiff } = tensor;
        let DispatchAutodiffContext::Enabled(strategy) = autodiff else {
            panic!("Requires autodiff tensor.");
        };
        let DispatchTensorKind::Autodiff(tensor) = kind else {
            panic!("Requires autodiff tensor.");
        };
        require_distributed_kind(&tensor.primitive().kind);

        let tensor = with_autodiff_backend!(strategy, |AD| {
            <AD as AutodiffBackend>::set_distributed_params(*tensor, param_id)
        });
        DispatchTensor {
            kind: DispatchTensorKind::Autodiff(Box::new(tensor)),
            autodiff,
        }
    }

    fn distributed_params(tensor: &DispatchTensor) -> Option<DistributedParams> {
        let DispatchTensorKind::Autodiff(autodiff) = &tensor.kind else {
            panic!("Requires autodiff tensor.");
        };
        let DispatchAutodiffContext::Enabled(strategy) = tensor.autodiff else {
            panic!("Requires autodiff tensor.");
        };
        with_autodiff_backend!(strategy, |AD| {
            <AD as AutodiffBackend>::distributed_params(autodiff)
        })
    }

    fn is_distributed(tensor: &DispatchTensor) -> bool {
        let DispatchTensorKind::Autodiff(autodiff) = &tensor.kind else {
            panic!("Requires autodiff tensor.");
        };
        let DispatchAutodiffContext::Enabled(strategy) = tensor.autodiff else {
            panic!("Requires autodiff tensor.");
        };
        with_autodiff_backend!(strategy, |AD| {
            <AD as AutodiffBackend>::is_distributed(autodiff)
        })
    }
}

#[cfg(all(test, feature = "autodiff", feature = "flex"))]
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
            expect_autodiff: bool,
        ) -> IntTensor<Self> {
            let _ = float;
            assert_eq!(expect_autodiff, B::ad_enabled(&int.device()));
            int
        }

        fn mixed_plain_float_and_enabled_int_route(
            float: FloatTensor<Self>,
            _int: IntTensor<Self>,
        ) -> FloatTensor<Self> {
            assert!(
                !B::ad_enabled(&float.device()),
                "an enabled integer should not route a plain float through autodiff"
            );
            float
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

        fn vector_float_route(
            int: IntTensor<Self>,
            _floats: Vec<FloatTensor<Self>>,
            expect_autodiff: bool,
        ) -> IntTensor<Self> {
            assert_eq!(expect_autodiff, B::ad_enabled(&int.device()));
            int
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
        let output = Dispatch::conditional_float_route(int, None, false);
        assert_eq!(output.autodiff, DispatchAutodiffContext::Enabled(strategy));

        let int = Dispatch::int_from_data(TensorData::from([1i32, 2]), &inner_device());
        let enabled_float = float([1.0, 2.0], &device(strategy));
        assert!(!Dispatch::is_tracked(&enabled_float));
        let output = Dispatch::conditional_float_route(int, Some(enabled_float), true);
        assert_eq!(output.autodiff, DispatchAutodiffContext::Enabled(strategy));

        let int = Dispatch::int_from_data(TensorData::from([1i32, 2]), &device(strategy));
        let plain_float = float([1.0, 2.0], &inner_device());
        let output = Dispatch::conditional_float_route(int, Some(plain_float), false);
        assert_eq!(output.autodiff, DispatchAutodiffContext::Enabled(strategy));
    }

    #[test]
    fn vector_float_route_uses_only_autodiff_floats_to_select_autodiff() {
        let strategy = GradientCheckpointingStrategy::Balanced;
        let int = Dispatch::int_from_data(TensorData::from([1i32, 2]), &device(strategy));
        let output =
            Dispatch::vector_float_route(int, vec![float([1.0, 2.0], &inner_device())], false);
        assert_eq!(output.autodiff, DispatchAutodiffContext::Enabled(strategy));

        let int = Dispatch::int_from_data(TensorData::from([1i32, 2]), &inner_device());
        let output = Dispatch::vector_float_route(
            int,
            vec![
                float([1.0, 2.0], &inner_device()),
                float([3.0, 4.0], &device(strategy)),
            ],
            true,
        );
        assert_eq!(output.autodiff, DispatchAutodiffContext::Enabled(strategy));
    }

    #[test]
    fn enabled_integer_does_not_route_a_plain_float_through_autodiff() {
        let strategy = GradientCheckpointingStrategy::Balanced;
        let float = float([1.0, 2.0], &inner_device());
        let int = Dispatch::int_from_data(TensorData::from([1i32, 2]), &device(strategy));

        let output = Dispatch::mixed_plain_float_and_enabled_int_route(float, int);

        assert_enabled_float(&output, strategy);
        assert!(!Dispatch::is_tracked(&output));
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
        // TODO: right now this assumes `type_id = 0`, but WgpuDevice has other types.
        #[allow(unreachable_patterns)]
        match type_id {
            #[cfg(cube_backend)]
            DispatchDeviceId::Cube => CubeDevice::enumerate_all()
                .into_iter()
                .filter(|device| cube_runtime_enabled(device.runtime()))
                .map(DispatchDevice::Cube)
                .collect(),
            #[cfg(feature = "flex")]
            DispatchDeviceId::Flex => vec![FlexDevice.into()],
            #[cfg(feature = "remote")]
            // A remote device needs its server's address, which a type id cannot carry.
            DispatchDeviceId::Remote => Vec::new(),
            #[cfg(feature = "shard")]
            // A sharded compute device is made from its members, which a type id cannot carry.
            DispatchDeviceId::ShardedCompute => Vec::new(),
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
            feature = "wgpu-msl",
            feature = "webgpu"
        )),
        RuntimeId::Cpu => cfg!(feature = "cpu"),
        // `metal` is cubecl's own Metal runtime; `wgpu-msl` is wgpu compiling to MSL, a
        // `RuntimeId::Wgpu` device.
        RuntimeId::Metal => cfg!(feature = "metal"),
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

#[cfg(all(test, feature = "autodiff", feature = "flex"))]
mod autodiff_representation {
    use super::*;
    use crate::GradientCheckpointingStrategy;
    #[cfg(feature = "wgpu")]
    use burn_backend::cubecl::Device as CubeDevice;
    use burn_backend::{
        AutodiffBackend, TensorData,
        ops::{AttentionModuleOptions, FloatTensorOps, IntTensorOps, ModuleOps, TransactionOps},
        tensor::FloatTensor,
    };

    type Ad = crate::backends::Autodiff<crate::Dispatch>;

    fn flex_device() -> DispatchDevice {
        DispatchDevice::autodiff(DispatchDevice::Flex(Default::default()))
    }

    fn wrap(tensor: FloatTensor<Ad>) -> DispatchTensor {
        DispatchTensor {
            kind: DispatchTensorKind::Autodiff(Box::new(tensor)),
            autodiff: DispatchAutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled),
        }
    }

    fn values(tensor: DispatchTensor) -> alloc::vec::Vec<f32> {
        let data = burn_backend::read_sync(Dispatch::float_into_data(tensor)).expect("data");
        data.as_slice::<f32>().expect("f32").to_vec()
    }

    /// Device queries must work on an autodiff device and on stacked autodiff devices.
    #[test]
    fn autodiff_devices_answer_backend_queries() {
        let device = flex_device();
        assert_eq!(Dispatch::name(&device), "dispatch<autodiff<flex>>");
        assert!(Dispatch::ad_enabled(&device));
        assert!(Dispatch::supports_dtype(&device, DType::F32));
        Dispatch::sync(&device).expect("autodiff device sync");

        let nested = DispatchDevice::autodiff(device.clone());
        assert_eq!(
            Dispatch::name(&nested),
            "dispatch<autodiff<autodiff<flex>>>"
        );
        assert!(Dispatch::supports_dtype(&nested, DType::F32));
        Dispatch::sync(&nested).expect("stacked autodiff device sync");
    }

    /// Reading a tracked tensor returns its value and leaves its graph usable.
    #[test]
    fn tracked_tensor_data_read_is_unchanged() {
        let x = Dispatch::float_from_data(TensorData::from([2.0f32, 3.0]), &flex_device());
        let x = Dispatch::float_set_require_grad(x, true);
        let y = Dispatch::float_mul(x.clone(), x.clone());

        assert_eq!(values(y.clone()), [4.0, 9.0]);

        let grads = Dispatch::backward(y);
        let grad = Dispatch::grad(&x, &grads).expect("gradient");
        assert_eq!(values(grad), [4.0, 6.0]);
    }

    /// Flex has no distributed support: the queries answer, and setting params still panics.
    #[test]
    fn flex_autodiff_tensor_has_no_distributed_params() {
        let x = Dispatch::float_from_data(TensorData::from([1.0f32]), &flex_device());
        let x = Dispatch::float_set_require_grad(x, true);

        assert!(Dispatch::distributed_params(&x).is_none());
        assert!(!Dispatch::is_distributed(&x));
    }

    #[test]
    #[should_panic(expected = "Distributed operations are not supported for tensor kind")]
    fn flex_autodiff_tensor_cannot_set_distributed_params() {
        let x = Dispatch::float_from_data(TensorData::from([1.0f32]), &flex_device());
        let x = Dispatch::float_set_require_grad(x, true);

        Dispatch::set_distributed_params(x, 0u64.into());
    }

    /// A transaction reads tracked floats by value without touching their graph.
    #[test]
    fn transaction_reads_tracked_floats() {
        use burn_backend::ops::TransactionPrimitive;

        let tracked = Dispatch::float_from_data(TensorData::from([1.0f32, 2.0]), &flex_device());
        let tracked = Dispatch::float_set_require_grad(tracked, true);
        let tracked = Dispatch::float_mul(tracked.clone(), tracked.clone());
        let plain = Dispatch::float_from_data(
            TensorData::from([3.0f32]),
            &DispatchDevice::Flex(Default::default()),
        );
        let int = Dispatch::int_from_data(TensorData::from([7i32]), &flex_device());

        let transaction = TransactionPrimitive::new(
            alloc::vec![tracked, plain],
            alloc::vec![],
            alloc::vec![int],
            alloc::vec![],
        );
        let data = burn_backend::read_sync(Dispatch::tr_execute(transaction)).expect("transaction");

        assert_eq!(
            data.read_floats[0].as_slice::<f32>().expect("f32"),
            [1.0, 4.0]
        );
        assert_eq!(data.read_floats[1].as_slice::<f32>().expect("f32"), [3.0]);
        assert_eq!(data.read_ints[0].as_slice::<i32>().expect("i32"), [7]);
    }

    /// The generated extraction rejects a tracked kind whose context says it is not associated.
    #[test]
    #[should_panic(expected = "an autodiff float primitive must have an enabled autodiff context")]
    fn autodiff_input_with_disabled_context_panics() {
        let tracked = Dispatch::float_from_data(TensorData::from([1.0f32]), &flex_device());
        let tracked = Dispatch::float_set_require_grad(tracked, true);
        let malformed = DispatchTensor {
            kind: DispatchTensorKind::Autodiff(Box::new(<Ad as AutodiffBackend>::from_inner(
                Dispatch::float_from_data(
                    TensorData::from([2.0f32]),
                    &DispatchDevice::Flex(Default::default()),
                ),
            ))),
            autodiff: DispatchAutodiffContext::Disabled,
        };

        Dispatch::float_add(tracked, malformed);
    }

    /// Every autodiff level is stripped when reading a stacked tensor's value.
    #[test]
    fn stacked_tensor_data_read_is_unchanged() {
        let device = flex_device();
        let x = Dispatch::float_from_data(TensorData::from([2.0f32, 3.0]), &device);
        let x = Dispatch::float_set_require_grad(x, true);
        let d1 = wrap(<Ad as AutodiffBackend>::from_inner(x));
        let d2 = wrap(<Ad as AutodiffBackend>::from_inner(d1.clone()).require_grad());

        let y = Dispatch::float_mul(d2.clone(), d2.clone());
        assert_eq!(values(y), [4.0, 9.0]);

        let grads = Dispatch::backward(Dispatch::float_mul(d2.clone(), d2.clone()));
        let first = Dispatch::grad(&d2, &grads).expect("first derivative");
        assert!(matches!(first.kind, DispatchTensorKind::Autodiff(_)));
    }

    // --- Nested autodiff (HOD) through the decomposed attention/CTC path ---

    type NestedInner = crate::backends::Autodiff<crate::Dispatch>;
    type NestedOuter = crate::backends::Autodiff<NestedInner>;

    fn nested_leaf<const D: usize>(
        values: &[f32],
        shape: [usize; D],
        device: &DispatchDevice,
    ) -> (FloatTensor<NestedInner>, FloatTensor<NestedOuter>) {
        let leaf = Dispatch::float_from_data(TensorData::new(values.to_vec(), shape), device);
        let inner = <NestedInner as AutodiffBackend>::from_inner(leaf);
        let inner = NestedInner::float_set_require_grad(inner, true);
        let outer = <NestedOuter as AutodiffBackend>::from_inner(inner.clone());
        let outer = NestedOuter::float_set_require_grad(outer, true);
        (inner, outer)
    }

    fn nested_attention_hod(device: DispatchDevice) {
        let shape = [1, 1, 2, 2];
        let q_values = [0.1f32, -0.2, 0.3, 0.4];
        let k_values = [0.2f32, 0.1, -0.3, 0.25];
        let v_values = [0.5f32, -0.1, 0.2, 0.3];
        let options = AttentionModuleOptions::default();

        let (qi, q) = nested_leaf(&q_values, shape, &device);
        let (_ki, k) = nested_leaf(&k_values, shape, &device);
        let (_vi, v) = nested_leaf(&v_values, shape, &device);

        let out = NestedOuter::attention(q.clone(), k, v, None, None, None, options);
        let g1 = NestedOuter::backward(out);
        let dq = NestedOuter::grad(&q, &g1).expect("first-order attention gradient");
        let g2 = NestedInner::backward(dq.clone());
        let d2 = NestedInner::grad(&qi, &g2).expect("second-order attention gradient");
        assert!(
            values(d2).iter().all(|v| v.is_finite()),
            "attention second derivative is finite"
        );

        // First-order comparison against the single-level autodiff backend.
        let leaf = Dispatch::float_from_data(TensorData::new(q_values.to_vec(), shape), &device);
        let single = <Ad as AutodiffBackend>::from_inner(leaf);
        let single = Ad::float_set_require_grad(single, true);
        let k_s = <Ad as AutodiffBackend>::from_inner(Dispatch::float_from_data(
            TensorData::new(k_values.to_vec(), shape),
            &device,
        ));
        let v_s = <Ad as AutodiffBackend>::from_inner(Dispatch::float_from_data(
            TensorData::new(v_values.to_vec(), shape),
            &device,
        ));
        let out_s = Ad::attention(single.clone(), k_s, v_s, None, None, None, options);
        let g_s = Ad::backward(out_s);
        let dq_s = Ad::grad(&single, &g_s).expect("single-level attention gradient");

        let nested_first = values(NestedInner::inner(dq.clone()));
        let single_first = values(dq_s);
        for (a, b) in nested_first.iter().zip(single_first.iter()) {
            assert!((a - b).abs() < 1e-4, "nested {a} vs single {b}");
        }
    }

    #[test]
    fn nested_attention_hod_on_flex() {
        nested_attention_hod(DispatchDevice::Flex(Default::default()));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn nested_attention_hod_on_wgpu() {
        nested_attention_hod(DispatchDevice::Cube(CubeDevice::Wgpu(Default::default())));
    }

    fn nested_ctc_hod(device: DispatchDevice) {
        let t = 3usize;
        let c = 3usize;
        let mut data = Vec::with_capacity(t * c);
        for time in 0..t {
            let logits = [0.2 * time as f32, -0.5, 0.1];
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for e in exps {
                data.push((e / sum).ln());
            }
        }

        let (log_probs_inner, log_probs) = nested_leaf(&data, [t, 1, c], &device);
        let targets = Dispatch::int_from_data(TensorData::from([[1i32]]), &device);
        let input_lengths = Dispatch::int_from_data(TensorData::from([t as i32]), &device);
        let target_lengths = Dispatch::int_from_data(TensorData::from([1i32]), &device);

        let loss = NestedOuter::float_sum(NestedOuter::ctc_loss(
            log_probs.clone(),
            targets,
            input_lengths,
            target_lengths,
            0,
        ));
        let g1 = NestedOuter::backward(loss);
        let dl = NestedOuter::grad(&log_probs, &g1).expect("first-order CTC gradient");
        let g2 = NestedInner::backward(dl.clone());
        let d2 = NestedInner::grad(&log_probs_inner, &g2).expect("second-order CTC gradient");
        assert!(
            values(d2).iter().all(|v| v.is_finite()),
            "CTC second derivative is finite"
        );
    }

    #[test]
    fn nested_ctc_hod_on_flex() {
        nested_ctc_hod(DispatchDevice::Flex(Default::default()));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn nested_ctc_hod_on_wgpu() {
        nested_ctc_hod(DispatchDevice::Cube(CubeDevice::Wgpu(Default::default())));
    }
}

/// The device under an autodiff wrapper, which a measurement of the device itself runs on.
fn measured(device: &DispatchDevice) -> &DispatchDevice {
    match device {
        #[cfg(feature = "autodiff")]
        DispatchDevice::Autodiff(inner) => &inner.inner,
        other => other,
    }
}
