use burn_ir::BackendIr;
use burn_tensor::Tensor;
use serde::{Deserialize, Serialize};

use crate::local_server::{LocalCollectiveClient, get_collective_client};

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum AggregateStrategy {
    Centralized,
    Tree(u32),
    Ring,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum AggregateKind {
    Sum,
    Mean,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct GlobalAggregateParams;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct AggregateParams {
    pub kind: AggregateKind,
    pub strategy: AggregateStrategy,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct GlobalRegisterParams {
    pub num_nodes: u32,
    pub server_address: String,
    pub client_address: String,
    pub client_data_port: u16,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct RegisterParams {
    pub num_local_nodes: u32,
    pub global_params: Option<GlobalRegisterParams>,
}

/// Resets the local collective server. All registered callers and ongoing operations are forgotten
pub fn reset_collective<B: BackendIr>() {
    let client = get_collective_client::<B>();
    client.reset();
}

/// Registers a "node". `num_nodes` must be the same as the other calls to register,
/// and `id` must be unique.
pub fn register<B: BackendIr>(id: u32, params: RegisterParams) {
    let client = get_collective_client::<B>();
    client.register(id, params);
}

/// Calls for an all-reduce operation with the given parameters, and returns the result.
/// The `params` must be the same as the parameters passed by the other nodes.
pub fn all_reduce<B: BackendIr, const D: usize>(
    tensor: Tensor<B, D>,
    params: AggregateParams,
) -> Tensor<B, D> {
    let client: LocalCollectiveClient<B> = get_collective_client();
    let device = tensor.device();
    let tensor = tensor.into_primitive().tensor();
    let primitive = client.aggregate(tensor, params);
    Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive)).to_device(&device)
}
