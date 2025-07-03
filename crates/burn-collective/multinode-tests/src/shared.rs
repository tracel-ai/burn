use burn::{backend::collective::AllReduceParams, tensor::TensorData};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct NodeTestData {
    pub device_count: u32,
    pub node_id: u32,
    pub node_count: u32,
    pub server_url: String,
    pub client_url: String,
    pub client_data_port: u16,

    pub aggregate_params: AllReduceParams,

    pub inputs: Vec<TensorData>,
    pub expected: TensorData,
}
