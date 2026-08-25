//! Non-executing router channel used to capture Burn operation graphs.

use alloc::{boxed::Box, collections::BTreeMap, format, string::String, sync::Arc, vec::Vec};
use burn_backend::{
    BoolStore, DType, DTypeUsage, DTypeUsageSet, DeviceId, DeviceOps, DeviceSettings,
    ExecutionError, Shape, TensorData,
};
use burn_ir::{
    GraphBindings, GraphId, GraphIr, IrVisitorMut, OperationIr, TensorId, TensorIr, TensorStatus,
};
use burn_std::{device::Device, future::DynFut, tensor::quantization::QuantConfig};
use hashbrown::{HashMap, HashSet};
use portable_atomic::{AtomicU64, Ordering};
use spin::Mutex;

use burn_router::{
    Graph, MultiBackendBridge, RouterChannel, RouterClient, RouterClientRegistration, RouterTensor,
    register_scoped_client,
};

static DEVICE_COUNTER: AtomicU64 = AtomicU64::new(0);
static TENSOR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Capture's backend-local device type identifier.
///
/// Dispatch reserves only the low eight bits for a backend's own type identifier, so this value
/// must remain representable as a `u8` for `DispatchDevice` ID round trips.
const CAPTURE_DEVICE_TYPE_ID: u16 = u8::MAX as u16;

/// Backend type that records operations instead of executing them.
pub type CaptureBackend = burn_router::BackendRouter<CaptureChannel>;

/// Reusable logical device for isolated graph-capture scopes.
///
/// Device identifiers are deliberately monotonic and never recycled. This keeps stale tensor
/// handles from aliasing a newer capture device without requiring an identity registry. Since a
/// [`DeviceId`] has a `u16` backend-local index, creating more than 65,536 capture devices in one
/// process will panic. A device can be reused for any number of sequential capture scopes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureDevice {
    id: u16,
}

impl CaptureDevice {
    /// Capture one operation scope on this reusable device.
    ///
    /// The closure receives a [`CaptureScope`] and must return the token produced by
    /// [`CaptureScope::complete`], declaring both the runtime input and requested output tensor
    /// IDs. A fresh session is installed before the closure runs and is always unregistered
    /// afterward, including while unwinding from a panic. Only one scope may be active on a capture
    /// device at a time. Calling `complete` closes the session immediately, so the closure must not
    /// perform more tensor operations before returning the completion token.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError::AlreadyActive`] if another capture scope is already active on this
    /// device.
    pub fn capture_scope(
        &self,
        capture: impl FnOnce(CaptureScope) -> CompletedCaptureScope,
    ) -> Result<CapturedGraph, CaptureError> {
        let session = Arc::new(CaptureSession::default());
        let client = CaptureClient::new(*self, session.clone());
        // Bind this reusable device to the new session client for exactly this scope. BackendRouter
        // operations call `get_client`, which now finds this client instead of trying the channel's
        // intentionally unsupported unscoped initialization path.
        let registration = register_scoped_client::<CaptureChannel>(self, client.clone())
            .ok_or(CaptureError::AlreadyActive)?;
        let guard = CaptureScopeGuard {
            client,
            _registration: registration,
        };
        guard.complete(capture(CaptureScope { session }))
    }
}

impl Default for CaptureDevice {
    fn default() -> Self {
        let id = DEVICE_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            id: u16::try_from(id).expect("capture device identifier space exhausted"),
        }
    }
}

impl Device for CaptureDevice {
    fn from_id(device_id: DeviceId) -> Self {
        assert_eq!(
            device_id.type_id, CAPTURE_DEVICE_TYPE_ID,
            "invalid capture device type"
        );
        Self {
            id: device_id.index_id,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: CAPTURE_DEVICE_TYPE_ID,
            index_id: self.id,
        }
    }
}

impl DeviceOps for CaptureDevice {
    fn defaults(&self) -> DeviceSettings {
        DeviceSettings::new(
            DType::F32,
            DType::I64,
            DType::Bool(BoolStore::U8),
            QuantConfig::default(),
        )
    }
}

/// Final result of a capture scope.
#[derive(Clone, Debug)]
pub struct CapturedGraph {
    /// Ordered operation graph with caller-declared boundaries.
    pub graph: GraphIr,
    /// Concrete values supplied through tensor initialization, keyed by tensor ID.
    pub values: BTreeMap<TensorId, TensorData>,
}

/// Token passed to a [`CaptureDevice::capture_scope`] closure.
///
/// Call [`complete`](Self::complete) after recording the operations in the scope. The closure's
/// return type requires the resulting completion token, so a successful capture cannot omit its
/// explicit graph boundaries.
#[derive(Debug)]
pub struct CaptureScope {
    // The scope owns the authority to close its session. Keeping this private prevents callers
    // from manufacturing completion tokens detached from the active router registration.
    session: Arc<CaptureSession>,
}

impl CaptureScope {
    /// Complete the scope with ordered runtime input and graph output tensor IDs.
    ///
    /// Module parameters and other initialized constants should not be listed as runtime inputs;
    /// their retained tensor values remain available in [`CapturedGraph::values`].
    /// The active session is closed before this returns; subsequent operations through tensors or
    /// the capture device are rejected even if the closure has not returned yet.
    pub fn complete(
        self,
        inputs: impl IntoIterator<Item = TensorId>,
        outputs: impl IntoIterator<Item = TensorId>,
    ) -> CompletedCaptureScope {
        let inputs = inputs.into_iter().collect();
        let outputs = outputs.into_iter().collect();
        self.session.state.lock().close();
        CompletedCaptureScope { inputs, outputs }
    }
}

/// Completion token required from a [`CaptureDevice::capture_scope`] closure.
///
/// This value can only be created by [`CaptureScope::complete`]. It carries the explicit graph
/// boundaries back to the capture device for validation and finalization.
#[derive(Debug)]
pub struct CompletedCaptureScope {
    inputs: Vec<TensorId>,
    outputs: Vec<TensorId>,
}

#[derive(Debug, Default)]
struct CaptureSession {
    state: Mutex<CaptureState>,
}

/// Owns the session client and its router binding for one capture scope.
///
/// The registration makes every router lookup for the device resolve to `client`. Dropping this
/// guard removes that lookup entry so the reusable device can be bound to a fresh session later.
struct CaptureScopeGuard {
    client: CaptureClient,
    // Owns the global router entry until completion, error, or panic unwinding.
    _registration: RouterClientRegistration<CaptureChannel>,
}

impl CaptureScopeGuard {
    fn complete(self, scope: CompletedCaptureScope) -> Result<CapturedGraph, CaptureError> {
        let mut state = self.client.state().lock();
        let inputs: Vec<_> = scope
            .inputs
            .into_iter()
            .map(|id| state.resolve_alias(id))
            .collect();
        let outputs: Vec<_> = scope
            .outputs
            .into_iter()
            .map(|id| state.resolve_alias(id))
            .collect();
        let known: HashSet<_> = state
            .operations
            .iter()
            .flat_map(OperationIr::nodes)
            .map(|tensor| tensor.id)
            .chain(state.values.keys().copied())
            .collect();
        for (boundary, ids) in [("input", &inputs), ("output", &outputs)] {
            let mut unique = HashSet::new();
            if let Some(id) = ids.iter().find(|id| !unique.insert(**id)) {
                return Err(CaptureError::DuplicateBoundary {
                    boundary,
                    tensor: *id,
                });
            }
            if let Some(id) = ids.iter().find(|id| !known.contains(*id)) {
                return Err(CaptureError::UnknownBoundary {
                    boundary,
                    tensor: *id,
                });
            }
        }

        let inferred = GraphIr::classify(&state.operations);
        for &id in &inputs {
            if !inferred.inputs.contains(&id) && !state.values.contains_key(&id) {
                return Err(CaptureError::InvalidInput { tensor: id });
            }
        }
        for &id in &inferred.inputs {
            if !inputs.contains(&id) && !state.values.contains_key(&id) {
                return Err(CaptureError::UndeclaredInput { tensor: id });
            }
        }
        for &id in &outputs {
            // Returning a graph input directly is valid. A computed output, however, must not have
            // been consumed in place or explicitly dropped according to boundary classification.
            let consumed = state.operations.iter().any(|operation| {
                matches!(operation, OperationIr::Drop(tensor) if tensor.id == id)
                    || operation
                        .nodes()
                        .into_iter()
                        .any(|tensor| tensor.id == id && tensor.status == TensorStatus::ReadWrite)
            });
            if consumed || (!inferred.inputs.contains(&id) && !inferred.outputs.contains(&id)) {
                return Err(CaptureError::InvalidOutput { tensor: id });
            }
        }
        let operations = core::mem::take(&mut state.operations);
        // Tensor handles can outlive the scope (for example, module running states shared across
        // clones). Keep initialized data in the closed session so those handles can still be
        // materialized onto another device without allowing any further graph operations.
        let values = state.values.clone();
        let captured = CapturedGraph {
            graph: GraphIr {
                operations,
                inputs,
                outputs,
            },
            values,
        };
        drop(state);
        Ok(captured)
    }
}

impl Drop for CaptureScopeGuard {
    fn drop(&mut self) {
        // Also close scopes that unwind before returning a completion token. Tensor handles can
        // retain a clone of the session client after the router registration itself is removed.
        self.client.state().lock().close();
    }
}

/// Error returned while finalizing a capture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptureError {
    /// Another capture scope is already active on the device.
    AlreadyActive,
    /// Capture was requested through a user-facing device using another backend.
    InvalidDevice,
    /// A declared boundary tensor was never initialized or referenced.
    UnknownBoundary {
        /// Whether this is an input or output boundary.
        boundary: &'static str,
        /// Unknown tensor ID.
        tensor: TensorId,
    },
    /// A tensor was declared more than once at the same boundary.
    DuplicateBoundary {
        /// Whether this is an input or output boundary.
        boundary: &'static str,
        /// Repeated tensor ID.
        tensor: TensorId,
    },
    /// A tensor dependency was neither initialized nor declared as a runtime input.
    UndeclaredInput {
        /// Tensor read by the graph without an available source.
        tensor: TensorId,
    },
    /// A declared input is produced within the graph rather than supplied externally.
    InvalidInput {
        /// Invalid input tensor ID.
        tensor: TensorId,
    },
    /// A declared output is consumed in place or otherwise cannot survive the graph.
    InvalidOutput {
        /// Invalid output tensor ID.
        tensor: TensorId,
    },
}

impl core::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AlreadyActive => write!(f, "a capture scope is already active on this device"),
            Self::InvalidDevice => write!(f, "capture scope requires a capture device"),
            Self::UnknownBoundary { boundary, tensor } => {
                write!(f, "unknown graph {boundary} tensor {tensor}")
            }
            Self::DuplicateBoundary { boundary, tensor } => {
                write!(f, "duplicate graph {boundary} tensor {tensor}")
            }
            Self::UndeclaredInput { tensor } => {
                write!(f, "graph reads undeclared input tensor {tensor}")
            }
            Self::InvalidInput { tensor } => {
                write!(
                    f,
                    "graph input tensor {tensor} is produced within the graph"
                )
            }
            Self::InvalidOutput { tensor } => {
                write!(f, "graph output tensor {tensor} does not survive the graph")
            }
        }
    }
}

/// Router channel selecting the non-executing capture client.
///
/// Capture clients are scope-specific: [`CaptureDevice::capture_scope`] creates the client and
/// installs it with `register_scoped_client` before any tensor operations run. An unscoped client
/// would have no lifecycle owner or completed graph boundary, so it is intentionally unsupported.
#[derive(Clone)]
pub struct CaptureChannel;

impl RouterChannel for CaptureChannel {
    type Device = CaptureDevice;
    type Bridge = CaptureBridge;
    type Client = CaptureClient;

    fn name(_device: &Self::Device) -> String {
        "capture".into()
    }

    fn init_client(_device: &Self::Device) -> Self::Client {
        // `get_client` reaches this only when no capture scope registered its client first.
        panic!("capture tensor operations must run inside CaptureDevice::capture_scope")
    }

    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> TensorData {
        client
            .state()
            .lock()
            .value(tensor.id)
            .unwrap_or_else(|| panic!("capture tensor {} has no initialized value", tensor.id))
    }

    fn register_tensor(
        client: &Self::Client,
        handle: TensorData,
        _shape: Shape,
        _dtype: DType,
    ) -> RouterTensor<Self::Client> {
        client.register_tensor_data(handle)
    }
}

/// Bridge for materialized values between independent capture devices.
///
/// Only initialized tensors have handles in the capture backend, so values reaching this bridge
/// are concrete and can safely initialize a tensor in another capture scope. Computed tensors have
/// no materialized handle and are rejected by [`CaptureChannel::get_tensor_handle`] first.
pub struct CaptureBridge;

impl MultiBackendBridge for CaptureBridge {
    type TensorHandle = TensorData;
    type Device = CaptureDevice;

    fn change_backend_float(
        tensor: TensorData,
        _shape: Shape,
        _target: &Self::Device,
    ) -> TensorData {
        tensor
    }
    fn change_backend_int(tensor: TensorData, _shape: Shape, _target: &Self::Device) -> TensorData {
        tensor
    }
    fn change_backend_bool(
        tensor: TensorData,
        _shape: Shape,
        _target: &Self::Device,
    ) -> TensorData {
        tensor
    }
}

#[derive(Debug)]
struct CaptureState {
    operations: Vec<OperationIr>,
    values: BTreeMap<TensorId, TensorData>,
    graphs: HashMap<GraphId, Graph>,
    aliases: HashMap<TensorId, TensorId>,
    closed: bool,
}

impl Default for CaptureState {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
            values: BTreeMap::new(),
            graphs: HashMap::new(),
            aliases: HashMap::new(),
            closed: false,
        }
    }
}

impl CaptureState {
    fn close(&mut self) {
        self.closed = true;
    }

    fn assert_open(&self) {
        assert!(!self.closed, "capture scope is already complete");
    }

    /// Resolve a fusion alias to the tensor ID that owns the computation or initializer.
    fn resolve_alias(&self, mut id: TensorId) -> TensorId {
        while let Some(source) = self.aliases.get(&id) {
            id = *source;
        }
        id
    }

    fn value(&self, id: TensorId) -> Option<TensorData> {
        self.values.get(&self.resolve_alias(id)).cloned()
    }

    /// Record a computation after replacing runtime-only fusion aliases with graph dependencies.
    ///
    /// Router `Drop` operations only manage handle lifetimes. Excluding them keeps the captured
    /// computation independent of incidental Rust cloning and ensures closure-local boundary
    /// tensors may be dropped before the scope is finalized.
    fn register_op(&mut self, mut op: OperationIr) {
        if matches!(op, OperationIr::Drop(_)) {
            return;
        }
        self.assert_open();
        op.visit_mut(&mut AliasVisitor {
            aliases: &self.aliases,
        });
        self.operations.push(op);
    }
}

struct AliasVisitor<'a> {
    aliases: &'a HashMap<TensorId, TensorId>,
}

impl IrVisitorMut for AliasVisitor<'_> {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        while let Some(source) = self.aliases.get(&tensor.id) {
            tensor.id = *source;
        }
    }
}

/// Client that records router operations and locally retains initialized values.
#[derive(Clone)]
pub struct CaptureClient {
    device: CaptureDevice,
    session: Arc<CaptureSession>,
}

impl CaptureClient {
    fn new(device: CaptureDevice, session: Arc<CaptureSession>) -> Self {
        Self { device, session }
    }

    fn state(&self) -> &Mutex<CaptureState> {
        &self.session.state
    }
}

impl RouterClient for CaptureClient {
    type Device = CaptureDevice;

    fn register_op(&self, op: OperationIr) {
        self.state().lock().register_op(op);
    }

    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        let value = self.state().lock().value(tensor.id);
        Box::pin(async move {
            value.ok_or_else(|| ExecutionError::WithContext {
                reason: format!("captured tensor {} has no concrete value", tensor.id),
            })
        })
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        Ok(())
    }
    fn flush(&self) {}

    fn create_empty_handle(&self) -> TensorId {
        self.state().lock().assert_open();
        TensorId::new(TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let mut state = self.state().lock();
        state.assert_open();
        let id = TensorId::new(TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed));
        let shape = data.shape.clone();
        let dtype = data.dtype;
        state.values.insert(id, data);
        drop(state);
        RouterTensor::new(id, shape, dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        self.device
    }

    fn seed(&self, _seed: u64) {
        panic!("seeding is not supported during graph capture")
    }

    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet {
        match dtype {
            // Capture records these operations without executing dtype-specific kernels. The
            // router's quantized operations are not implemented yet, so quantized tensors remain
            // the only dtype family that capture cannot represent through the backend API.
            DType::QFloat(_) => DTypeUsageSet::empty(),
            _ => DTypeUsage::general(),
        }
    }

    fn register_and_execute_graph(
        &self,
        graph_id: GraphId,
        relative_graph: Vec<OperationIr>,
        bindings: GraphBindings,
    ) {
        let graph = Graph::new(relative_graph);
        let bound = graph.bind(bindings);
        let mut state = self.state().lock();
        state.graphs.insert(graph_id, graph);
        for operation in bound.operations {
            state.register_op(operation);
        }
    }

    fn execute_graph(&self, graph_id: GraphId, bindings: GraphBindings) {
        let graph = self
            .state()
            .lock()
            .graphs
            .get(&graph_id)
            .cloned()
            .unwrap_or_else(|| panic!("capture graph {graph_id:?} was not registered"));
        let bound = graph.bind(bindings);
        let mut state = self.state().lock();
        for operation in bound.operations {
            state.register_op(operation);
        }
    }

    fn register_alias(&self, new_id: TensorId, src_id: TensorId) {
        let mut state = self.state().lock();
        state.assert_open();
        let source = state.resolve_alias(src_id);
        if source != new_id {
            state.aliases.insert(new_id, source);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    #[cfg(feature = "std")]
    use burn_backend::Backend;
    use burn_backend::ops::FloatTensorOps;
    use burn_ir::{CustomOpIr, ScalarIr};
    use burn_router::get_client;

    fn tensor(id: u64, shape: impl Into<Shape>) -> TensorIr {
        TensorIr::uninit(TensorId::new(id), shape.into(), DType::F32)
    }

    #[test]
    fn backend_router_operations_are_captured() {
        let device = CaptureDevice::default();
        let captured = device
            .capture_scope(|scope| {
                let lhs = CaptureBackend::float_from_data(TensorData::from([1.0f32, 2.0]), &device);
                let rhs = CaptureBackend::float_from_data(TensorData::from([3.0f32, 4.0]), &device);
                let lhs_id = lhs.id();
                let rhs_id = rhs.id();
                let output = CaptureBackend::float_add(lhs, rhs);
                let output_id = output.id();

                scope.complete([lhs_id, rhs_id], [output_id])
            })
            .unwrap();
        assert_eq!(captured.values.len(), 2);
        assert!(
            captured.graph.operations.iter().any(|operation| matches!(
                operation,
                OperationIr::NumericFloat(_, burn_ir::NumericOperationIr::Add(_))
            )),
            "captured operations: {:?}",
            captured.graph.operations
        );
        assert!(
            captured
                .graph
                .operations
                .iter()
                .all(|operation| !matches!(operation, OperationIr::Drop(_))),
            "tensor lifetime operations must not be part of a computation graph"
        );
    }

    #[test]
    fn captures_operations_values_and_explicit_boundaries() {
        let device = CaptureDevice::default();
        let captured = device
            .capture_scope(|scope| {
                let client = get_client::<CaptureChannel>(&device);
                let input = client.register_tensor_data(TensorData::from([1.0f32, 2.0]));
                let input_id = input.id();
                let input_ir = input.into_ir();
                let output_id = client.create_empty_handle();
                let output_ir = TensorIr::uninit(output_id, Shape::new([2]), DType::F32);
                client.register_op(OperationIr::Custom(CustomOpIr::new(
                    "identity",
                    &[input_ir],
                    &[output_ir],
                )));

                scope.complete([input_id], [output_id])
            })
            .unwrap();
        let input_id = captured.graph.inputs[0];
        let output_id = captured.graph.outputs[0];
        assert_eq!(captured.graph.inputs, [input_id]);
        assert_eq!(captured.graph.outputs, [output_id]);
        assert_eq!(captured.graph.operations.len(), 1);
        assert_eq!(
            captured.values.get(&input_id),
            Some(&TensorData::from([1.0f32, 2.0]))
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn escaped_tensors_cannot_record_after_their_scope_completes() {
        let device = CaptureDevice::default();
        let mut escaped_output = None;
        let first = device
            .capture_scope(|scope| {
                let lhs = CaptureBackend::float_from_data(TensorData::from([1.0f32]), &device);
                let rhs = CaptureBackend::float_from_data(TensorData::from([2.0f32]), &device);
                let lhs_id = lhs.id();
                let rhs_id = rhs.id();
                let output = CaptureBackend::float_add(lhs, rhs);
                let output_id = output.id();
                escaped_output = Some(output);

                scope.complete([lhs_id, rhs_id], [output_id])
            })
            .unwrap();
        assert!(!first.graph.operations.is_empty());

        let next = device
            .capture_scope(|scope| {
                let fresh = CaptureBackend::float_from_data(TensorData::from([3.0f32]), &device);
                let fresh_id = fresh.id();
                let stale = escaped_output.take().unwrap();
                let operation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    CaptureBackend::float_add(stale, fresh)
                }));

                assert!(operation.is_err());
                scope.complete([fresh_id], [])
            })
            .unwrap();
        assert_eq!(next.graph.operations.len(), 1);
        assert!(matches!(next.graph.operations[0], OperationIr::Init(_)));
    }

    #[test]
    fn initialized_tensor_can_move_from_completed_scope_to_new_device() {
        let first_device = CaptureDevice::default();
        let second_device = CaptureDevice::default();
        let mut escaped = None;

        first_device
            .capture_scope(|scope| {
                let tensor =
                    CaptureBackend::float_from_data(TensorData::from([1.0f32, 2.0]), &first_device);
                let tensor_id = tensor.id();
                escaped = Some(tensor);
                scope.complete([tensor_id], [])
            })
            .unwrap();

        let captured = second_device
            .capture_scope(|scope| {
                let tensor =
                    CaptureBackend::float_to_device(escaped.take().unwrap(), &second_device);
                let output = CaptureBackend::float_neg(tensor);
                let output_id = output.id();
                scope.complete([], [output_id])
            })
            .unwrap();

        assert_eq!(captured.values.len(), 1);
        assert!(matches!(
            captured.graph.operations.last(),
            Some(OperationIr::NumericFloat(
                _,
                burn_ir::NumericOperationIr::Neg(_)
            ))
        ));
    }

    #[cfg(feature = "std")]
    #[test]
    fn completing_scope_immediately_rejects_more_operations() {
        let device = CaptureDevice::default();
        let captured = device
            .capture_scope(|scope| {
                let client = get_client::<CaptureChannel>(&device);
                let completed = scope.complete([], []);
                let tensor_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    CaptureBackend::float_from_data(TensorData::from([1.0f32]), &device)
                }));
                let handle = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    client.create_empty_handle()
                }));
                let operation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    client.register_op(OperationIr::Custom(CustomOpIr::new(
                        "late",
                        &[],
                        &[tensor(99_000, [1])],
                    )))
                }));

                assert!(tensor_result.is_err());
                assert!(handle.is_err());
                assert!(operation.is_err());
                completed
            })
            .unwrap();

        assert!(captured.graph.operations.is_empty());
    }

    #[test]
    fn capture_device_can_be_reused_for_sequential_scopes() {
        let device = CaptureDevice::default();
        let first = device
            .capture_scope(|scope| scope.complete([], []))
            .unwrap();
        let second = device
            .capture_scope(|scope| scope.complete([], []))
            .unwrap();

        assert!(first.graph.operations.is_empty());
        assert!(second.graph.operations.is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn tensor_operations_require_an_active_scope() {
        let device = CaptureDevice::default();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = CaptureBackend::float_from_data(TensorData::from([1.0f32]), &device);
        }));

        assert!(panic.is_err());
        assert!(device.capture_scope(|scope| scope.complete([], [])).is_ok());
    }

    #[test]
    fn overlapping_scopes_on_one_device_are_rejected() {
        let device = CaptureDevice::default();
        let outer = device.capture_scope(|scope| {
            let inner = device.capture_scope(|scope| scope.complete([], []));
            assert!(matches!(inner, Err(CaptureError::AlreadyActive)));
            scope.complete([], [])
        });

        assert!(outer.is_ok());
    }

    #[test]
    fn separate_devices_support_overlapping_scopes() {
        let first = CaptureDevice::default();
        let second = CaptureDevice::default();

        let outer = first.capture_scope(|outer| {
            let inner = second
                .capture_scope(|inner| inner.complete([], []))
                .unwrap();
            assert!(inner.graph.operations.is_empty());
            outer.complete([], [])
        });

        assert!(outer.is_ok());
    }

    #[cfg(feature = "std")]
    #[test]
    fn panicking_scope_releases_its_router_registration() {
        let device = CaptureDevice::default();
        let mut escaped = None;
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = device.capture_scope(|_| -> CompletedCaptureScope {
                escaped = Some(CaptureBackend::float_from_data(
                    TensorData::from([1.0f32]),
                    &device,
                ));
                panic!("capture failed")
            });
        }));

        assert!(panic.is_err());
        let stale = escaped.unwrap();
        let operation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CaptureBackend::float_neg(stale)
        }));
        assert!(operation.is_err());
        assert!(device.capture_scope(|scope| scope.complete([], [])).is_ok());
    }

    #[test]
    fn capture_devices_have_distinct_monotonic_ids() {
        let first = CaptureDevice::default();
        let second = CaptureDevice::default();

        assert_ne!(first, second);
        assert!(first.to_id().index_id < second.to_id().index_id);
    }

    #[test]
    fn capture_device_id_round_trips() {
        let device = CaptureDevice::default();
        let restored = CaptureDevice::from_id(device.to_id());
        assert_eq!(restored, device);
        assert_eq!(restored.to_id().type_id, CAPTURE_DEVICE_TYPE_ID);
    }

    #[test]
    fn invalid_boundary_is_reported() {
        let device = CaptureDevice::default();
        let captured = device.capture_scope(|scope| {
            let client = get_client::<CaptureChannel>(&device);
            let _tensor = client.register_tensor_data(TensorData::from([3i64]));
            scope.complete([TensorId::new(u64::MAX)], [])
        });

        assert!(matches!(
            captured,
            Err(CaptureError::UnknownBoundary { .. })
        ));
    }

    #[test]
    fn duplicate_boundary_is_reported() {
        let device = CaptureDevice::default();
        let mut duplicate = None;
        let captured = device.capture_scope(|scope| {
            let client = get_client::<CaptureChannel>(&device);
            let tensor = client.register_tensor_data(TensorData::from([3i64]));
            let id = tensor.id();
            duplicate = Some(id);
            scope.complete([id, id], [])
        });

        assert!(matches!(
            captured,
            Err(CaptureError::DuplicateBoundary { tensor, .. }) if Some(tensor) == duplicate
        ));
    }

    #[test]
    fn dangling_graph_input_is_reported() {
        let device = CaptureDevice::default();
        let captured = device.capture_scope(|scope| {
            let client = get_client::<CaptureChannel>(&device);
            let missing = tensor(90_000, [1]);
            let output_id = client.create_empty_handle();
            client.register_op(OperationIr::Custom(CustomOpIr::new(
                "dangling",
                &[missing],
                &[tensor(output_id.value(), [1])],
            )));
            scope.complete([], [output_id])
        });

        assert!(matches!(
            captured,
            Err(CaptureError::UndeclaredInput { .. })
        ));
    }

    #[test]
    fn graph_produced_tensor_cannot_be_declared_as_input() {
        let device = CaptureDevice::default();
        let captured = device.capture_scope(|scope| {
            let client = get_client::<CaptureChannel>(&device);
            let input = client.register_tensor_data(TensorData::from([1.0f32]));
            let output_id = client.create_empty_handle();
            client.register_op(OperationIr::Custom(CustomOpIr::new(
                "produce",
                &[input.into_ir()],
                &[tensor(output_id.value(), [1])],
            )));
            scope.complete([output_id], [output_id])
        });

        assert!(matches!(captured, Err(CaptureError::InvalidInput { .. })));
    }

    #[test]
    fn read_write_tensor_cannot_be_declared_as_output() {
        let device = CaptureDevice::default();
        let captured = device.capture_scope(|scope| {
            let client = get_client::<CaptureChannel>(&device);
            let input = client.register_tensor_data(TensorData::from([1.0f32]));
            let input_id = input.id();
            let mut input = input.into_ir();
            input.status = TensorStatus::ReadWrite;
            let output_id = client.create_empty_handle();
            client.register_op(OperationIr::Custom(CustomOpIr::new(
                "consume_in_place",
                &[input],
                &[tensor(output_id.value(), [1])],
            )));
            scope.complete([input_id], [input_id])
        });

        assert!(matches!(captured, Err(CaptureError::InvalidOutput { .. })));
    }

    #[test]
    fn capture_supports_every_non_quantized_dtype() {
        let device = CaptureDevice::default();
        let captured = device.capture_scope(|scope| {
            let client = get_client::<CaptureChannel>(&device);
            for dtype in [
                DType::F64,
                DType::F32,
                DType::Flex32,
                DType::F16,
                DType::BF16,
                DType::I64,
                DType::I32,
                DType::I16,
                DType::I8,
                DType::U64,
                DType::U32,
                DType::U16,
                DType::U8,
                DType::Bool(BoolStore::Native),
                DType::Bool(BoolStore::U8),
                DType::Bool(BoolStore::U32),
            ] {
                assert_eq!(
                    client.dtype_usage(dtype),
                    DTypeUsage::general(),
                    "{dtype:?}"
                );
            }
            assert!(
                client
                    .dtype_usage(DType::QFloat(
                        burn_backend::quantization::QuantScheme::default()
                    ))
                    .is_empty()
            );
            scope.complete([], [])
        });

        assert!(captured.is_ok());
    }

    #[cfg(feature = "std")]
    #[test]
    fn seeding_capture_device_is_rejected() {
        let device = CaptureDevice::default();
        let captured = device.capture_scope(|scope| {
            let seeded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                CaptureBackend::seed(&device, 42)
            }));
            assert!(seeded.is_err());
            scope.complete([], [])
        });

        assert!(captured.is_ok());
    }

    #[test]
    fn self_alias_is_ignored() {
        let device = CaptureDevice::default();
        let captured = device.capture_scope(|scope| {
            let client = get_client::<CaptureChannel>(&device);
            let id = client.create_empty_handle();
            client.register_alias(id, id);
            assert_eq!(client.state().lock().resolve_alias(id), id);
            scope.complete([], [])
        });

        assert!(captured.is_ok());
    }

    #[test]
    fn cached_graph_invocations_are_bound_and_recorded() {
        let device = CaptureDevice::default();
        let input_1 = TensorId::new(10_000);
        let output_1 = TensorId::new(10_001);
        let input_2 = TensorId::new(20_000);
        let output_2 = TensorId::new(20_001);
        let captured = device
            .capture_scope(|scope| {
                let client = get_client::<CaptureChannel>(&device);
                let relative_input = tensor(0, Shape::new([0]));
                let relative_intermediate = tensor(1, Shape::new([0]));
                let relative_output = tensor(2, Shape::new([0]));
                let relative_graph = vec![
                    OperationIr::Custom(CustomOpIr::new(
                        "first",
                        core::slice::from_ref(&relative_input),
                        core::slice::from_ref(&relative_intermediate),
                    )),
                    OperationIr::Custom(CustomOpIr::with_scalars(
                        "second",
                        core::slice::from_ref(&relative_intermediate),
                        core::slice::from_ref(&relative_output),
                        vec![ScalarIr::UInt(0)],
                    )),
                ];
                let graph_id = GraphId(7);

                client.register_and_execute_graph(
                    graph_id,
                    relative_graph,
                    GraphBindings {
                        tensors: vec![(relative_input.id, input_1), (relative_output.id, output_1)],
                        shapes: vec![3],
                        scalars: vec![ScalarIr::Float(1.5)],
                        ranges: vec![],
                    },
                );
                client.execute_graph(
                    graph_id,
                    GraphBindings {
                        tensors: vec![(relative_input.id, input_2), (relative_output.id, output_2)],
                        shapes: vec![5],
                        scalars: vec![ScalarIr::Float(2.5)],
                        ranges: vec![],
                    },
                );

                scope.complete([input_1, input_2], [output_1, output_2])
            })
            .unwrap();
        assert_eq!(captured.graph.operations.len(), 4);

        let OperationIr::Custom(first_1) = &captured.graph.operations[0] else {
            panic!("expected first custom operation")
        };
        let OperationIr::Custom(second_1) = &captured.graph.operations[1] else {
            panic!("expected second custom operation")
        };
        let OperationIr::Custom(first_2) = &captured.graph.operations[2] else {
            panic!("expected first custom operation")
        };
        let OperationIr::Custom(second_2) = &captured.graph.operations[3] else {
            panic!("expected second custom operation")
        };
        assert_eq!(first_1.inputs[0].id, input_1);
        assert_eq!(second_1.outputs[0].id, output_1);
        assert_eq!(first_1.outputs[0].id, second_1.inputs[0].id);
        assert_eq!(first_1.inputs[0].shape, Shape::new([3]));
        assert_eq!(second_1.scalars, [ScalarIr::Float(1.5)]);
        assert_eq!(first_2.inputs[0].id, input_2);
        assert_eq!(second_2.outputs[0].id, output_2);
        assert_eq!(first_2.outputs[0].id, second_2.inputs[0].id);
        assert_ne!(first_1.outputs[0].id, first_2.outputs[0].id);
        assert_eq!(first_2.inputs[0].shape, Shape::new([5]));
        assert_eq!(second_2.scalars, [ScalarIr::Float(2.5)]);
    }

    #[test]
    fn computed_fusion_aliases_resolve_to_their_source() {
        let device = CaptureDevice::default();
        let mut source = None;
        let mut alias = None;
        let captured = device
            .capture_scope(|scope| {
                let client = get_client::<CaptureChannel>(&device);
                let input = client.register_tensor_data(TensorData::from([1.0f32, 2.0]));
                let input_id = input.id();
                let source_id = client.create_empty_handle();
                let alias_id = client.create_empty_handle();
                let output_id = client.create_empty_handle();

                client.register_op(OperationIr::Custom(CustomOpIr::new(
                    "produce",
                    &[input.into_ir()],
                    &[TensorIr::uninit(source_id, Shape::new([2]), DType::F32)],
                )));
                client.register_alias(alias_id, source_id);
                client.register_op(OperationIr::Custom(CustomOpIr::new(
                    "consume",
                    &[TensorIr::uninit(alias_id, Shape::new([2]), DType::F32)],
                    &[TensorIr::uninit(output_id, Shape::new([2]), DType::F32)],
                )));
                source = Some(source_id);
                alias = Some(alias_id);

                scope.complete([input_id], [output_id])
            })
            .unwrap();
        let source_id = source.unwrap();
        let alias_id = alias.unwrap();
        let OperationIr::Custom(consume) = &captured.graph.operations[1] else {
            panic!("expected alias consumer")
        };
        assert_eq!(consume.inputs[0].id, source_id);
        assert!(!captured.graph.inputs.contains(&alias_id));
    }

    #[test]
    fn initialized_aliases_share_the_source_value_and_boundary() {
        let device = CaptureDevice::default();
        let mut source = None;
        let captured = device
            .capture_scope(|scope| {
                let client = get_client::<CaptureChannel>(&device);
                let tensor = client.register_tensor_data(TensorData::from([4i64]));
                let source_id = tensor.id();
                let alias_id = client.create_empty_handle();
                client.register_alias(alias_id, source_id);

                assert_eq!(
                    client.state().lock().value(alias_id),
                    Some(TensorData::from([4i64]))
                );
                source = Some(source_id);
                scope.complete([alias_id], [])
            })
            .unwrap();
        let source_id = source.unwrap();
        assert_eq!(captured.graph.inputs, [source_id]);
        assert!(captured.values.contains_key(&source_id));
    }
}
