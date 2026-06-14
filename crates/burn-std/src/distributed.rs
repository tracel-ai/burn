use serde::{Deserialize, Serialize};

/// The different ways to execute the reduce operation.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize)]
pub enum ReduceOperation {
    /// The sum of the values.
    Sum,
    /// The mean of the values.
    Mean,
}

/// Parameter struct for setting up and getting parameters for distributed operations.
#[derive(Clone, Debug)]
pub struct DistributedConfig {
    /// How to execute the all_reduce operation.
    pub all_reduce_op: ReduceOperation,
}
