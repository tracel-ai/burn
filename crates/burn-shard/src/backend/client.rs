use burn_backend::RandomKey;
use std::{any::Any, sync::Arc};

use burn_backend::{
    DType, DTypeUsageSet, DeviceSettings, EnvironmentBundle, EnvironmentError, EnvironmentId,
    EnvironmentOrigin, EnvironmentSummary, ExecutionError, ExecutionMode, ExecutionModeToken,
    PruneOptions, Pruned, SaveOptions, TensorData,
};
use burn_ir::{
    BackendIr, CapturedGraphId, GraphBindings, GraphId, HandleKind, OperationIr, TensorId, TensorIr,
};
use burn_router::{RouterClient, RouterTensor};
use burn_std::{future::DynFut, sync::Mutex};

use super::{ShardedComputeDevice, interpreter::DynShardInterpreter};
use crate::{Sharding, Shards, ShardsError};

/// The client of one sharded compute device: every tensor of the sharded compute device goes through its interpreter.
#[derive(Clone)]
pub struct ShardClient {
    device: ShardedComputeDevice,
    interpreter: Arc<Mutex<dyn DynShardInterpreter>>,
    settings: DeviceSettings,
    /// The members' devices, in member order, as a `Vec` of their backend's device.
    members: Arc<dyn Any + Send + Sync>,
}

impl ShardClient {
    pub fn new<D: Any + Send + Sync>(
        device: ShardedComputeDevice,
        interpreter: impl DynShardInterpreter + 'static,
        settings: DeviceSettings,
        members: Vec<D>,
    ) -> Self {
        Self {
            device,
            interpreter: Arc::new(Mutex::new(interpreter)),
            settings,
            members: Arc::new(members),
        }
    }

    /// The members' devices, in member order, when they are `D`.
    pub fn members<D: Any + Clone>(&self) -> Option<Vec<D>> {
        self.members.downcast_ref::<Vec<D>>().cloned()
    }

    pub fn settings(&self) -> DeviceSettings {
        self.settings
    }

    /// The tensor `shards` make up, one shard per member of the members' backend `B`, each
    /// already on its member.
    ///
    /// # Errors
    ///
    /// When the members run another backend than `B`, or the shards do not make up one tensor.
    pub fn join_shards<B: BackendIr>(
        &self,
        shards: Shards<HandleKind<B>>,
    ) -> Result<RouterTensor<Self>, ShardsError> {
        let joined = self.interpreter.lock().register_shards(Box::new(shards))?;
        Ok(RouterTensor::new(
            joined.id,
            joined.shape,
            joined.dtype,
            self.clone(),
        ))
    }

    /// Each member's shard of `tensor`, in member order, and how they make up its value.
    ///
    /// # Panics
    ///
    /// When the members run another backend than `B`.
    pub fn take_shards<B: BackendIr>(&self, tensor: RouterTensor<Self>) -> Shards<HandleKind<B>> {
        // Giving up the tensor can register a drop through this same interpreter, so its ir is
        // taken before the lock.
        let ir = tensor.into_ir();
        let shards = self.interpreter.lock().take_shards(ir);
        *shards
            .downcast::<Shards<HandleKind<B>>>()
            .unwrap_or_else(|_| panic!("The members do not run {}", core::any::type_name::<B>()))
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

    fn environment_save(
        &self,
        _id: EnvironmentId,
        _options: SaveOptions,
    ) -> Result<EnvironmentBundle, EnvironmentError> {
        Err(shard_environment())
    }

    fn environment_summary(
        &self,
        _id: EnvironmentId,
    ) -> Result<EnvironmentSummary, EnvironmentError> {
        Err(shard_environment())
    }

    fn environment_merge(
        &self,
        _id: EnvironmentId,
        _origin: EnvironmentOrigin,
    ) -> Result<(), EnvironmentError> {
        Err(shard_environment())
    }

    fn environment_prune(
        &self,
        _id: EnvironmentId,
        _options: PruneOptions,
    ) -> Result<Pruned, EnvironmentError> {
        Err(shard_environment())
    }

    fn graph_prepare(&self) -> Result<(), ExecutionError> {
        self.interpreter.lock().graph_prepare()
    }

    fn graph_start_capture(&self) -> Result<(), ExecutionError> {
        self.interpreter.lock().graph_start_capture()
    }

    fn graph_stop_capture(&self) -> Result<CapturedGraphId, ExecutionError> {
        self.interpreter.lock().graph_stop_capture()
    }

    unsafe fn graph_replay(&self, graph: CapturedGraphId) -> Result<(), ExecutionError> {
        // Safety: forwarded verbatim from this method's own contract.
        unsafe { self.interpreter.lock().graph_replay(graph) }
    }

    fn graph_release(&self, graph: CapturedGraphId) {
        self.interpreter.lock().graph_release(graph);
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
