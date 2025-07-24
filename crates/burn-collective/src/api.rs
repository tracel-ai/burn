use burn_tensor::{Tensor, backend::Backend};

use crate::{
    CollectiveConfig, PeerId, ReduceOperation, global::shared::GlobalCollectiveError,
    local_server::get_collective_client,
};

/// Errors from collective operations
#[allow(unused)]
#[derive(Debug, Clone)]
pub enum CollectiveError {
    /// The [config](CollectiveConfig) was invalid.
    /// Usually happens if only some global parameters have been defined
    InvalidConfig,
    /// Cannot un-register a node twice
    MultipleUnregister,
    /// Cannot register a node twice
    MultipleRegister,
    /// Trying to register a different way than is currently being done
    RegisterParamsMismatch,
    /// Trying to all-reduce a different way than is currently being done: op must match
    AllReduceParamsMismatch,
    /// Trying to reduce a different way than is currently being done: root and op must match
    ReduceParamsMismatch,
    /// Trying to broadcast a different way than is currently being done:
    /// root must match and only one tensor
    BroadcastParamsMismatch,
    /// Trying to broadcast but no peer sent a tensor
    /// TODO see if broadcast needs root as an argument or not
    BroadcastNoTensor,
    /// Local collective server couldn't respond
    LocalServerMissing,
    /// Another operation was called before Register
    RegisterNotFirstOperation,
    /// The global orchestrator had an error
    Global(GlobalCollectiveError),

    #[allow(unused)]
    Other(String),
}

/// Registers a device. `num_devices` must be the same for every register,
/// and `device_id` must be unique.
///
/// * `id` - The peer id of the caller
///
/// With auto-diff backends, make sure to use the inner backend.
pub fn register<B: Backend>(
    id: PeerId,
    device: B::Device,
    config: CollectiveConfig,
) -> Result<(), CollectiveError> {
    let mut client = get_collective_client::<B>();
    client.register(id, device, config)
}

/// Calls for an all-reduce operation with the given parameters, and returns the result.
/// The `params` must be the same as the parameters passed by the other nodes.
///
/// * `id` - The peer id of the caller
/// * `tensor` - The input tensor to reduce with the peers' tensors
/// * `config` - Config of the collective operation, must be coherent with the other calls
pub fn all_reduce<B: Backend, const D: usize>(
    id: PeerId,
    tensor: Tensor<B, D>,
    op: ReduceOperation,
) -> Result<Tensor<B, D>, CollectiveError> {
    let client = get_collective_client::<B>();
    client.all_reduce(id, tensor, op)
}

/// Broadcasts, or recives a broadcasted tensor.
///
/// * `id` - The peer id of the caller
/// * `tensor` - If defined, this tensor will be broadcasted. Otherwise, this call will receive
///   the broadcasted tensor.
/// * `root` - The peer that will broadcast the tensor.
/// * `config` - Config of the collective operation, must be coherent with the other calls
///
/// Returns the broadcasted tensor.
pub fn broadcast<B: Backend, const D: usize>(
    id: PeerId,
    tensor: Option<Tensor<B, D>>,
    root: PeerId,
) -> Result<Tensor<B, D>, CollectiveError> {
    let client = get_collective_client::<B>();
    client.broadcast(id, tensor, root)
}

/// Reduces a tensor onto one device.
///
/// * `id` - The peer id of the caller
/// * `tensor` - The tensor to send as input
/// * `root` - The ID of the peer that will receive the result.
/// * `config` - Config of the collective operation, must be coherent with the other calls
///
/// Returns Ok(None) if the root tensor is not the caller. Otherwise, returns the reduced tensor.
pub fn reduce<B: Backend, const D: usize>(
    id: PeerId,
    tensor: Tensor<B, D>,
    op: ReduceOperation,
    root: PeerId,
) -> Result<Option<Tensor<B, D>>, CollectiveError> {
    let client = get_collective_client::<B>();
    client.reduce(id, tensor, op, root)
}

/// Closes the collective session, unregistering the device
pub fn finish_collective<B: Backend>(id: PeerId) -> Result<(), CollectiveError> {
    let client = get_collective_client::<B>();
    client.finish(id)
}

/// Resets the local collective server. All registered callers and ongoing operations are forgotten
pub fn reset_collective<B: Backend>() {
    let client = get_collective_client::<B>();
    client.reset();
}
