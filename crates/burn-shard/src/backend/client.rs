use burn_backend::RandomKey;
use std::sync::Arc;

use burn_backend::{
    DType, DTypeUsageSet, DeviceSettings, EnvironmentBundle, EnvironmentError, EnvironmentId,
    EnvironmentOrigin, ExecutionError, ExecutionMode, ExecutionModeToken, TensorData,
    TensorMetadata,
};
use burn_ir::{GraphBindings, GraphId, OperationIr, TensorId, TensorIr};
use burn_router::{RouterClient, RouterTensor};
use burn_std::{future::DynFut, sync::Mutex};

use super::{ShardedComputeDevice, interpreter::DynShardInterpreter};
use crate::{Sharding, ShardingError};

/// The client of one sharded compute device: every tensor of the sharded compute device goes through its interpreter.
#[derive(Clone)]
pub struct ShardClient {
    device: ShardedComputeDevice,
    interpreter: Arc<Mutex<dyn DynShardInterpreter>>,
    settings: DeviceSettings,
}

impl ShardClient {
    pub fn new(
        device: ShardedComputeDevice,
        interpreter: impl DynShardInterpreter + 'static,
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

    /// How many members the tensors are split over.
    pub fn num_members(&self) -> usize {
        self.interpreter.lock().num_members()
    }
}

impl RouterClient for ShardClient {
    type Device = ShardedComputeDevice;

    fn register_op(&self, op: OperationIr) {
        self.interpreter.lock().register_op(op);
    }

    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        self.interpreter.lock().read(tensor)
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        self.interpreter.lock().sync()
    }

    fn flush(&self) -> Result<(), ExecutionError> {
        self.interpreter.lock().flush()
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

    fn next_random_key(&self) -> RandomKey {
        self.interpreter.lock().next_random_key()
    }

    fn seed(&self, seed: u64) {
        self.interpreter.lock().seed(seed);
    }

    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet {
        self.interpreter.lock().dtype_usage(dtype)
    }

    /// Refused rather than answered `None`: the sharded compute device's members may well apply a mode, and a
    /// caller told that the sharded compute device does not would run a compile-only pass for real.
    fn execution_mode_open(
        &self,
        mode: ExecutionMode,
    ) -> Result<Option<ExecutionModeToken>, ExecutionError> {
        Err(ExecutionError::with_context(format!(
            "a sharded compute device does not open {mode:?}: open it on the sharded compute device's devices"
        )))
    }

    /// A sharded compute device's members each keep their own environment: create and activate one on them.
    fn environment_create(
        &self,
        _name: String,
        _origin: EnvironmentOrigin,
    ) -> Result<EnvironmentId, EnvironmentError> {
        Err(shard_environment())
    }

    fn environment_activate(&self, _id: EnvironmentId) -> Result<EnvironmentId, EnvironmentError> {
        Err(shard_environment())
    }

    fn environment_active(&self) -> Result<EnvironmentId, EnvironmentError> {
        Err(shard_environment())
    }

    fn environment_save(&self, _id: EnvironmentId) -> Result<EnvironmentBundle, EnvironmentError> {
        Err(shard_environment())
    }

    fn environment_merge(
        &self,
        _id: EnvironmentId,
        _origin: EnvironmentOrigin,
    ) -> Result<(), EnvironmentError> {
        Err(shard_environment())
    }

    fn register_and_execute_graph(&self, _: GraphId, _: Vec<OperationIr>, _: GraphBindings) {
        unreachable!(
            "Only router fusion replays graphs, and a sharded compute device runs without it"
        )
    }

    fn execute_graph(&self, _: GraphId, _: GraphBindings) {
        unreachable!(
            "Only router fusion replays graphs, and a sharded compute device runs without it"
        )
    }

    fn register_alias(&self, _: TensorId, _: TensorId) {
        unreachable!(
            "Only router fusion aliases handles, and a sharded compute device runs without it"
        )
    }
}

/// The answer of a sharded compute device to an environment call.
fn shard_environment() -> EnvironmentError {
    EnvironmentError::IncompatibleDevice {
        reason: "a sharded compute device keeps no environment of its own: use its devices'".into(),
    }
}
