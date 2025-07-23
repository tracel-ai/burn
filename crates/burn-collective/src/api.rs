use burn_tensor::{backend::Backend, Tensor};

use crate::{
    global::shared::GlobalCollectiveError, local_server::get_collective_client, CollectiveConfig, DeviceId, ReduceOperation
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
    /// Trying to aggregate a different way than is currently being done
    AllReduceParamsMismatch,
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
    id: DeviceId,
    config: CollectiveConfig,
) -> Result<(), CollectiveError> {
    let mut client = get_collective_client::<B>();
    client.register(id, config)
}

/// Calls for an all-reduce operation with the given parameters, and returns the result.
/// The `params` must be the same as the parameters passed by the other nodes.
///
/// * `id` - The peer id of the caller
/// * `tensor` - The input tensor to reduce with the peers' tensors
/// * `config` - Config of the collective operation, must be coherent with the other calls
pub fn all_reduce<B: Backend, const D: usize>(
    id: DeviceId,
    tensor: Tensor<B, D>,
    op: ReduceOperation,
) -> Result<Tensor<B, D>, CollectiveError> {
    let client = get_collective_client::<B>();
    let device = tensor.device();
    let tensor = tensor.into_primitive().tensor();
    let primitive = client.all_reduce(id, tensor, op)?;
    let tensor =
        Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive)).to_device(&device);

    Ok(tensor)
}

/// Broadcasts, or recives a broadcasted tensor.
///
/// * `id` - The peer id of the caller
/// * `tensor` - If defined, this tensor will be broadcasted. Otherwise, this call will receive
///     the broadcasted tensor.
/// * `root` - The peer that will broadcast the tensor.
/// * `config` - Config of the collective operation, must be coherent with the other calls
///
/// Returns the broadcasted tensor.
pub fn broadcast<B: Backend, const D: usize>(
    id: DeviceId,
    tensor: Option<Tensor<B, D>>,
    _device: B::Device, // TODO `register` should return a client, and collective ops should be done on the client.
    root: DeviceId,
) -> Result<Tensor<B, D>, CollectiveError> {
    let client = get_collective_client::<B>();
    let tensor = tensor.map(|tensor| {
        tensor.device();
        tensor.into_primitive().tensor()
    });
    let primitive = client.broadcast(id, tensor, root)?;
    let tensor =
        Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive));

    Ok(tensor)
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
    id: DeviceId,
    tensor: Tensor<B, D>,
    op: ReduceOperation,
    root: DeviceId,
) -> Result<Option<Tensor<B, D>>, CollectiveError> {
    let client = get_collective_client::<B>();
    let device = tensor.device();
    let tensor = tensor.into_primitive().tensor();
    let primitive = client.reduce(id, tensor, op, root)?;
    let tensor = primitive.map(|primitive| {
        Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive)).to_device(&device)
    });

    Ok(tensor)
}

/// Closes the collective session, unregistering the device
pub fn finish_collective<B: Backend>(id: DeviceId) -> Result<(), CollectiveError> {
    let client = get_collective_client::<B>();
    client.finish(id)
}

/// Resets the local collective server. All registered callers and ongoing operations are forgotten
pub fn reset_collective<B: Backend>() {
    let client = get_collective_client::<B>();
    client.reset();
}
