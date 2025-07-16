use burn::tensor::TensorData;
use burn_collective::{SharedAllReduceParams, global::shared::base::NodeId};
use burn_network::network::NetworkAddress;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct NodeTestData {
    pub device_count: u32,
    pub node_id: NodeId,
    pub node_count: u32,
    pub server_address: NetworkAddress,
    pub client_address: NetworkAddress,
    pub client_data_port: u16,

    pub aggregate_params: SharedAllReduceParams,

    pub inputs: Vec<TensorData>,
    pub expected: TensorData,
}
