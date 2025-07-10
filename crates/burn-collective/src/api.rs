use burn_network::network::NetworkAddress;
use burn_tensor::{Tensor, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{
    global::server::base::GlobalCollectiveError,
    local_server::{LocalCollectiveClient, get_collective_client},
};

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum AllReduceStrategy {
    Centralized,
    Tree(u32),
    Ring,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ReduceKind {
    Sum,
    Mean,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct GlobalAllReduceParams {
    pub strategy: AllReduceStrategy,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct AllReduceParams {
    pub kind: ReduceKind,
    pub local_strategy: AllReduceStrategy,
    pub global_strategy: Option<AllReduceStrategy>,
}

#[allow(unused)]
#[derive(Debug)]
pub(crate) enum CollectiveError {
    // Cannot un-register a node twice
    MultipleUnregister,
    // Cannot register a node twice
    MultipleRegister,
    // Trying to register a different way than is currently being done
    RegisterParamsMismatch,
    // Trying to aggregate a different way than is currently being done
    AllReduceParamsMismatch,
    // Couldn't send a callback to a client
    CallbackFailed,
    // The Global collective server had an error
    Global(GlobalCollectiveError),

    // TODO handle case where register is not the first operation!
    #[allow(unused)]
    Other(String),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct GlobalRegisterParams {
    /// The id of this node, should be unique.
    pub node_id: u32,
    /// The number of nodes globally. Should be the same for all nodes.
    pub num_nodes: u32,
    /// The address for the connection to this client.
    pub server_address: NetworkAddress,
    /// The address for the connection to this client.
    pub client_address: NetworkAddress,
    /// The port on which to open the tensor data service for other clients. Should match the port
    /// given in the client url.
    pub client_data_port: u16,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct RegisterParams {
    pub num_devices: u32,
    pub global_params: Option<GlobalRegisterParams>,
}

/// Resets the local collective server. All registered callers and ongoing operations are forgotten
pub fn reset_collective<B: Backend>() {
    let client = get_collective_client::<B>();
    client.reset();
}

/// Registers a "node". `num_nodes` must be the same as the other calls to register,
/// and `device_id` must be unique.
pub fn register<B: Backend>(device_id: u32, params: RegisterParams) {
    let mut client = get_collective_client::<B>();
    client.register(device_id, params);
}

/// Calls for an all-reduce operation with the given parameters, and returns the result.
/// The `params` must be the same as the parameters passed by the other nodes.
pub fn all_reduce<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    params: AllReduceParams,
) -> Tensor<B, D> {
    let client: LocalCollectiveClient<B> = get_collective_client();
    let device = tensor.device();
    let tensor = tensor.into_primitive().tensor();
    let primitive = client.all_reduce(tensor, params);
    Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive)).to_device(&device)
}

pub fn finish_collective<B: Backend>(id: u32) {
    let client: LocalCollectiveClient<B> = get_collective_client();
    client.finish(id);
}
