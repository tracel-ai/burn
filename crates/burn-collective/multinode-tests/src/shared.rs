use burn::tensor::TensorData;
use burn_collective::{AllReduceStrategy, NodeId, ReduceKind};
use burn_communication::Address;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTestData {
    /// How many threads to start on this node
    pub device_count: u32,
    /// ID for this node
    pub node_id: NodeId,
    /// How many nodes in the cluster
    pub node_count: u32,
    /// Global server address
    pub global_address: Address,
    /// Node address
    pub node_address: Address,
    /// Node's data service port, for initializing the p2p tensor data service
    pub data_service_port: u16,
    /// What kind of aggregation
    pub all_reduce_kind: ReduceKind,
    /// Node's data service port, for initializing the p2p tensor data service
    pub global_strategy: AllReduceStrategy,
    /// What kind of aggregation
    pub local_strategy: AllReduceStrategy,

    /// Input data for test
    pub inputs: Vec<TensorData>,
    /// Expected output for test
    pub expected: TensorData,
}
