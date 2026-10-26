use std::sync::Arc;

use burn_backend::{
    DType, DTypeUsageSet, DeviceSettings, ExecutionError, TensorData, TensorMetadata,
};
use burn_ir::{GraphBindings, GraphId, OperationIr, TensorId, TensorIr};
use burn_router::{RouterClient, RouterTensor};
use burn_std::{future::DynFut, sync::Mutex};

use super::{GroupDevice, interpreter::DynGroupInterpreter};
use crate::{Sharding, ShardingError};

/// The client of one device group: every tensor of the group goes through its interpreter.
#[derive(Clone)]
pub struct GroupClient {
    device: GroupDevice,
    interpreter: Arc<Mutex<dyn DynGroupInterpreter>>,
    settings: DeviceSettings,
}

impl GroupClient {
    pub fn new(
        device: GroupDevice,
        interpreter: impl DynGroupInterpreter + 'static,
        settings: DeviceSettings,
    ) -> Self {
        Self {
            device,
            interpreter: Arc::new(Mutex::new(interpreter)),
            settings,
        }
    }

    pub fn settings(&self) -> DeviceSettings {
        self.settings
    }

    /// The same value with another sharding, or why no resharding produces it.
    pub fn reshard(
        &self,
        tensor: RouterTensor<Self>,
        sharding: Sharding,
    ) -> Result<RouterTensor<Self>, ShardingError> {
        let (shape, dtype) = (tensor.shape(), tensor.dtype());
        // Giving up the tensor can register a drop through this same interpreter, so its ir is
        // taken before the lock.
        let ir = tensor.into_ir();
        let id = self.interpreter.lock().reshard(ir, sharding)?;
        Ok(RouterTensor::new(id, shape, dtype, self.clone()))
    }

    /// Where the tensor's shards sit.
    pub fn sharding(&self, tensor: &RouterTensor<Self>) -> Sharding {
        self.interpreter.lock().sharding(&tensor.id())
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

    fn register_and_execute_graph(&self, _: GraphId, _: Vec<OperationIr>, _: GraphBindings) {
        unreachable!("Only router fusion replays graphs, and a device group runs without it")
    }

    fn execute_graph(&self, _: GraphId, _: GraphBindings) {
        unreachable!("Only router fusion replays graphs, and a device group runs without it")
    }

    fn register_alias(&self, _: TensorId, _: TensorId) {
        unreachable!("Only router fusion aliases handles, and a device group runs without it")
    }
}
