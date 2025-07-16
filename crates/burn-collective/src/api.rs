use burn_network::network::NetworkAddress;
use burn_tensor::{Tensor, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{
    global::{server::base::GlobalCollectiveError, shared::base::NodeId},
    local_server::get_collective_client,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterParams {
    pub device_id: DeviceId,
    pub shared: SharedRegisterParams,
    pub global: Option<GlobalRegisterParams>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct SharedRegisterParams {
    pub num_devices: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalRegisterParams {
    /// The id of this node, should be unique.
    pub node_id: NodeId,
    /// The address for the connection to this client.
    pub server_address: NetworkAddress,
    /// The address for the connection to this client.
    pub client_address: NetworkAddress,
    /// The port on which to open the tensor data service for other clients. Should match the port
    /// given in the client url.
    pub client_data_port: u16,

    /// Parameters that should be shared between different nodes
    pub shared_params: SharedGlobalRegisterParams,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct SharedGlobalRegisterParams {
    /// The number of nodes globally.
    pub num_nodes: u32,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct SharedAllReduceParams {
    pub kind: ReduceKind,
    pub local_strategy: AllReduceStrategy,
    pub global_strategy: Option<AllReduceStrategy>,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ReduceKind {
    Sum,
    Mean,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum AllReduceStrategy {
    Centralized,
    Tree(u32),
    Ring,
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub enum CollectiveError {
    // Cannot un-register a node twice
    MultipleUnregister,
    // Cannot register a node twice
    MultipleRegister,
    // Trying to register a different way than is currently being done
    RegisterParamsMismatch,
    // Trying to aggregate a different way than is currently being done
    AllReduceParamsMismatch,
    // Local collective server couldn't respond
    LocalServerMissing,
    // Another operation was called before Register
    RegisterNotFirstOperation,
    // The Global collective server had an error
    Global(GlobalCollectiveError),

    #[allow(unused)]
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId(u32);

impl From<u32> for DeviceId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

/// Registers a "node". `num_nodes` must be the same as the other calls to register,
/// and `device_id` must be unique. Registering is done per Backend.
pub fn register<B: Backend>(params: RegisterParams) -> Result<(), CollectiveError> {
    let mut client = get_collective_client::<B>();
    client.register(params)
}

/// Calls for an all-reduce operation with the given parameters, and returns the result.
/// The `params` must be the same as the parameters passed by the other nodes.
pub fn all_reduce<B: Backend, const D: usize>(
    id: DeviceId,
    tensor: Tensor<B, D>,
    params: &SharedAllReduceParams,
) -> Result<Tensor<B, D>, CollectiveError> {
    let client = get_collective_client::<B>();
    let device = tensor.device();
    let tensor = tensor.into_primitive().tensor();
    let primitive = client.all_reduce(id, tensor, params)?;
    let tensor =
        Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive)).to_device(&device);

    eprintln!("All-Reduce done");
    Ok(tensor)
}

pub fn finish_collective<B: Backend>(id: DeviceId) -> Result<(), CollectiveError> {
    let client = get_collective_client::<B>();
    client.finish(id)
}

/// Resets the local collective server. All registered callers and ongoing operations are forgotten
pub fn reset_collective<B: Backend>() {
    let client = get_collective_client::<B>();
    client.reset();
}
