use std::time::Duration;

use burn::tensor::{TensorData, backend::ReduceOperation};
use burn_collective::{AllReduceStrategy, NodeId};
use burn_communication::Address;
use serde::{Deserialize, Serialize};

/// Ranks of inputs and outputs for all testing
pub const TENSOR_RANK: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTest {
    /// How many threads to start on this node
    pub device_count: usize,
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
    /// What kind of all-reduce
    pub all_reduce_op: ReduceOperation,
    /// Node's data service port, for initializing the p2p tensor data service
    pub global_strategy: AllReduceStrategy,
    /// What kind of aggregation
    pub local_strategy: AllReduceStrategy,

    /// Input data for test: all tensors are D=3
    pub inputs: Vec<TensorData>,
    /// Expected output for test
    pub expected: TensorData,
}

/// Result sent back from each node for each test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTestResult {
    pub success: bool,
    pub durations: Vec<Duration>,
}
