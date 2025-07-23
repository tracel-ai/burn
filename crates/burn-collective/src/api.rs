use burn_tensor::{Tensor, backend::Backend};

use crate::{
    CollectiveConfig, global::shared::GlobalCollectiveError, local_server::get_collective_client,
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
/// With auto-diff backends, make sure to use the inner backend.
pub fn register<B: Backend>(config: &CollectiveConfig) -> Result<(), CollectiveError> {
    let mut client = get_collective_client::<B>();
    client.register(config)
}

/// Calls for an all-reduce operation with the given parameters, and returns the result.
/// The `params` must be the same as the parameters passed by the other nodes.
pub fn all_reduce<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    config: &CollectiveConfig,
) -> Result<Tensor<B, D>, CollectiveError> {
    let client = get_collective_client::<B>();
    let device = tensor.device();
    let tensor = tensor.into_primitive().tensor();
    let primitive = client.all_reduce(tensor, config)?;
    let tensor =
        Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive)).to_device(&device);

    Ok(tensor)
}

/// Closes the collective session, unregistering the device
pub fn finish_collective<B: Backend>(config: &CollectiveConfig) -> Result<(), CollectiveError> {
    let client = get_collective_client::<B>();
    client.finish(config.device_id)
}

/// Resets the local collective server. All registered callers and ongoing operations are forgotten
pub fn reset_collective<B: Backend>() {
    let client = get_collective_client::<B>();
    client.reset();
}
