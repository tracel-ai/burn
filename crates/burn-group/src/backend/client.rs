use std::sync::Arc;

use burn_backend::{DType, DTypeUsageSet, ExecutionError, TensorData, TensorMetadata};
use burn_ir::{GraphBindings, GraphId, OperationIr, TensorId, TensorIr};
use burn_router::{RouterClient, RouterTensor};
use burn_std::{future::DynFut, sync::Mutex};

use super::{GroupDevice, executor::GroupInterpreter};
use crate::GroupPlacement;

/// The client of one device group: every tensor of the group goes through its interpreter.
#[derive(Clone)]
pub struct GroupClient {
    device: GroupDevice,
    interpreter: Arc<Mutex<Box<dyn GroupInterpreter>>>,
}

impl GroupClient {
    pub fn new(device: GroupDevice) -> Self {
        Self {
            device,
            interpreter: Arc::new(Mutex::new(device.interpreter())),
        }
    }

    /// The same value at another placement.
    ///
    /// # Panics
    ///
    /// When no collective produces `placement`: a partial sum out of a whole tensor, or a split
    /// along a dim shorter than the group.
    pub fn place(
        &self,
        tensor: RouterTensor<Self>,
        placement: GroupPlacement,
    ) -> RouterTensor<Self> {
        let (shape, dtype) = (tensor.shape(), tensor.dtype());
        let id = self.interpreter.lock().place(tensor.into_ir(), placement);
        RouterTensor::new(id, shape, dtype, self.clone())
    }

    /// Where the tensor's shards sit.
    pub fn placement(&self, tensor: &RouterTensor<Self>) -> GroupPlacement {
        self.interpreter.lock().placement(&tensor.id())
    }
}

impl RouterClient for GroupClient {
    type Device = GroupDevice;

    fn register_op(&self, op: OperationIr) {
        self.interpreter.lock().register_op(op);
    }

    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        self.interpreter.lock().read(tensor)
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        self.interpreter.lock().sync()
    }

    fn flush(&self) {
        self.interpreter.lock().flush();
    }

    fn create_empty_handle(&self) -> TensorId {
        self.interpreter.lock().new_tensor_id()
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let (shape, dtype) = (data.shape().clone(), data.dtype());
        let id = self.interpreter.lock().register_tensor_data(data);
        RouterTensor::new(id, shape, dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        self.device
    }

    fn seed(&self, seed: u64) {
        self.interpreter.lock().seed(seed);
    }

    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet {
        self.interpreter.lock().dtype_usage(dtype)
    }

    fn register_and_execute_graph(
        &self,
        graph_id: GraphId,
        relative_graph: Vec<OperationIr>,
        bindings: GraphBindings,
    ) {
        self.interpreter
            .lock()
            .register_graph(graph_id, relative_graph, bindings);
    }

    fn execute_graph(&self, graph_id: GraphId, bindings: GraphBindings) {
        self.interpreter.lock().execute_graph(graph_id, bindings);
    }

    fn register_alias(&self, new_id: TensorId, src_id: TensorId) {
        self.interpreter.lock().register_alias(new_id, src_id);
    }
}
