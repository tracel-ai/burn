//! Non-executing router channel used to capture Burn operation graphs.

use alloc::{boxed::Box, format, string::String, sync::Arc, vec::Vec};
use core::sync::atomic::{AtomicU64, Ordering};

use burn_backend::{
    BoolStore, DType, DTypeUsage, DTypeUsageSet, DeviceId, DeviceOps, DeviceSettings,
    ExecutionError, Shape, TensorData,
};
use burn_ir::{GraphBindings, GraphId, GraphIr, OperationIr, TensorId, TensorIr};
use burn_std::{device::Device, future::DynFut};
use hashbrown::{HashMap, HashSet};
use spin::Mutex;

use burn_router::{MultiBackendBridge, RouterChannel, RouterClient, RouterTensor, get_client};

static DEVICE_COUNTER: AtomicU64 = AtomicU64::new(0);
static TENSOR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Backend type that records operations instead of executing them.
pub type CaptureBackend = burn_router::BackendRouter<CaptureChannel>;

/// Isolated logical device associated with one graph-capture session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureDevice {
    id: u16,
}

impl CaptureDevice {
    /// Start a new isolated capture session.
    pub fn capture() -> (Self, GraphCapture) {
        let device = Self::default();
        let capture = GraphCapture {
            client: get_client::<CaptureChannel>(&device),
        };
        (device, capture)
    }
}

impl Default for CaptureDevice {
    fn default() -> Self {
        let id = DEVICE_COUNTER.fetch_add(1, Ordering::Relaxed);
        assert!(
            id <= u16::MAX as u64,
            "capture device identifier space exhausted"
        );
        Self { id: id as u16 }
    }
}

impl Device for CaptureDevice {
    fn from_id(device_id: DeviceId) -> Self {
        assert_eq!(device_id.type_id, u16::MAX, "invalid capture device type");
        Self {
            id: device_id.index_id,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: u16::MAX,
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
            Default::default(),
        )
    }
}

/// Final result of a capture session.
#[derive(Clone, Debug)]
pub struct CapturedGraph {
    /// Ordered operation graph with caller-declared boundaries.
    pub graph: GraphIr,
    /// Concrete values supplied through tensor initialization, keyed by tensor ID.
    pub values: HashMap<TensorId, TensorData>,
}

/// Handle controlling the lifecycle of a [`CaptureDevice`] session.
#[derive(Clone)]
pub struct GraphCapture {
    client: CaptureClient,
}

impl GraphCapture {
    /// Finish the current capture, drain its operations and values, and declare boundaries.
    ///
    /// The handle remains reusable: operations registered afterward belong to a fresh capture.
    pub fn finish(
        &self,
        inputs: impl IntoIterator<Item = TensorId>,
        outputs: impl IntoIterator<Item = TensorId>,
    ) -> Result<CapturedGraph, CaptureError> {
        let mut state = self.client.state.lock();
        let known: HashSet<_> = state
            .operations
            .iter()
            .flat_map(OperationIr::nodes)
            .map(|tensor| tensor.id)
            .chain(state.values.keys().copied())
            .collect();
        let inputs: Vec<_> = inputs.into_iter().collect();
        let outputs: Vec<_> = outputs.into_iter().collect();
        for (boundary, ids) in [("input", &inputs), ("output", &outputs)] {
            if let Some(id) = ids.iter().find(|id| !known.contains(*id)) {
                return Err(CaptureError::UnknownBoundary {
                    boundary,
                    tensor: *id,
                });
            }
        }
        let operations = core::mem::take(&mut state.operations);
        let values = core::mem::take(&mut state.values);
        Ok(CapturedGraph {
            graph: GraphIr {
                operations,
                inputs,
                outputs,
            },
            values,
        })
    }

    /// Discard all operations and initialized values recorded so far.
    pub fn reset(&self) {
        let mut state = self.client.state.lock();
        state.operations.clear();
        state.values.clear();
    }
}

/// Error returned while finalizing a capture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptureError {
    /// A declared boundary tensor was never initialized or referenced.
    UnknownBoundary {
        /// Whether this is an input or output boundary.
        boundary: &'static str,
        /// Unknown tensor ID.
        tensor: TensorId,
    },
}

impl core::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownBoundary { boundary, tensor } => {
                write!(f, "unknown graph {boundary} tensor {tensor}")
            }
        }
    }
}

/// Router channel selecting the non-executing capture client.
#[derive(Clone)]
pub struct CaptureChannel;

impl RouterChannel for CaptureChannel {
    type Device = CaptureDevice;
    type Bridge = CaptureBridge;
    type Client = CaptureClient;

    fn name(_device: &Self::Device) -> String {
        "capture".into()
    }

    fn init_client(device: &Self::Device) -> Self::Client {
        CaptureClient::new(*device)
    }

    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> TensorData {
        client
            .state
            .lock()
            .values
            .get(&tensor.id)
            .cloned()
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

/// Identity bridge for initialized values moved between capture devices.
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

#[derive(Default)]
struct CaptureState {
    operations: Vec<OperationIr>,
    values: HashMap<TensorId, TensorData>,
}

/// Client that records router operations and locally retains initialized values.
#[derive(Clone)]
pub struct CaptureClient {
    device: CaptureDevice,
    state: Arc<Mutex<CaptureState>>,
}

impl CaptureClient {
    fn new(device: CaptureDevice) -> Self {
        Self {
            device,
            state: Arc::new(Mutex::new(CaptureState::default())),
        }
    }
}

impl RouterClient for CaptureClient {
    type Device = CaptureDevice;

    fn register_op(&self, op: OperationIr) {
        self.state.lock().operations.push(op);
    }

    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        let value = self.state.lock().values.get(&tensor.id).cloned();
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
        TensorId::new(TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let id = self.create_empty_handle();
        let shape = data.shape.clone();
        let dtype = data.dtype;
        self.state.lock().values.insert(id, data);
        RouterTensor::new(id, shape, dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        self.device
    }
    fn seed(&self, _seed: u64) {}
    fn dtype_usage(&self, _dtype: DType) -> DTypeUsageSet {
        DTypeUsage::general()
    }

    fn register_and_execute_graph(
        &self,
        _graph_id: GraphId,
        relative_graph: Vec<OperationIr>,
        _bindings: GraphBindings,
    ) {
        self.state.lock().operations.extend(relative_graph);
    }

    fn execute_graph(&self, _graph_id: GraphId, _bindings: GraphBindings) {
        panic!("cached graph replay is not supported during graph capture")
    }

    fn register_alias(&self, new_id: TensorId, src_id: TensorId) {
        let mut state = self.state.lock();
        if let Some(value) = state.values.get(&src_id).cloned() {
            state.values.insert(new_id, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::ops::FloatTensorOps;
    use burn_ir::CustomOpIr;

    #[test]
    fn backend_router_operations_are_captured() {
        let (device, capture) = CaptureDevice::capture();
        let lhs = CaptureBackend::float_from_data(TensorData::from([1.0f32, 2.0]), &device);
        let rhs = CaptureBackend::float_from_data(TensorData::from([3.0f32, 4.0]), &device);
        let lhs_id = lhs.id();
        let rhs_id = rhs.id();
        let output = CaptureBackend::float_add(lhs, rhs);
        let output_id = output.id();
        let _output_ir = output.into_ir();

        let captured = capture.finish([lhs_id, rhs_id], [output_id]).unwrap();
        assert_eq!(captured.values.len(), 2);
        assert!(
            captured.graph.operations.iter().any(|operation| matches!(
                operation,
                OperationIr::NumericFloat(_, burn_ir::NumericOperationIr::Add(_))
            )),
            "captured operations: {:?}",
            captured.graph.operations
        );
    }

    #[test]
    fn captures_operations_values_and_explicit_boundaries() {
        let (device, capture) = CaptureDevice::capture();
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

        let captured = capture.finish([input_id], [output_id]).unwrap();
        assert_eq!(captured.graph.inputs, [input_id]);
        assert_eq!(captured.graph.outputs, [output_id]);
        assert_eq!(captured.graph.operations.len(), 1);
        assert_eq!(
            captured.values.get(&input_id),
            Some(&TensorData::from([1.0f32, 2.0]))
        );
    }

    #[test]
    fn finish_starts_a_fresh_capture_and_reset_discards_state() {
        let (device, capture) = CaptureDevice::capture();
        let client = get_client::<CaptureChannel>(&device);
        let tensor = client.register_tensor_data(TensorData::from([1i64]));
        let id = tensor.id();
        let first = capture.finish([id], []).unwrap();
        assert_eq!(first.values.len(), 1);

        let empty = capture.finish([], []).unwrap();
        assert!(empty.graph.operations.is_empty());
        assert!(empty.values.is_empty());

        let _tensor = client.register_tensor_data(TensorData::from([2i64]));
        capture.reset();
        let reset = capture.finish([], []).unwrap();
        assert!(reset.values.is_empty());
    }

    #[test]
    fn invalid_boundary_does_not_consume_capture() {
        let (device, capture) = CaptureDevice::capture();
        let client = get_client::<CaptureChannel>(&device);
        let tensor = client.register_tensor_data(TensorData::from([3i64]));
        let id = tensor.id();
        assert!(capture.finish([TensorId::new(u64::MAX)], []).is_err());
        assert!(capture.finish([id], []).unwrap().values.contains_key(&id));
    }
}
