use burn::tensor::TensorData;
use burn_collective::{NodeId, SharedAllReduceParams};
use burn_communication::Address;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct NodeTestData {
    pub device_count: u32,
    pub node_id: NodeId,
    pub node_count: u32,
    pub server_address: Address,
    pub client_address: Address,
    pub client_data_port: u16,

    pub aggregate_params: SharedAllReduceParams,

    pub inputs: Vec<TensorData>,
    pub expected: TensorData,
}
