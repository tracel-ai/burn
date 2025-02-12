use std::fmt::Display;

use burn_common::id::{IdGenerator, StreamId};
use burn_ir::{OperationIr, TensorId, TensorIr};
use burn_tensor::TensorData;
use serde::{Deserialize, Serialize};

#[allow(missing_docs)]
#[derive(new, Serialize, Deserialize, Debug, Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub struct ConnectionId {
    pub position: u64,
    pub stream_id: StreamId,
}

/// Unique identifier that can represent a session.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct SessionId {
    id: u64,
}

impl Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SessionId({})", self.id)
    }
}

impl SessionId {
    /// Create a new [session id](SessionId).
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            id: IdGenerator::generate(),
        }
    }
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub enum Task {
    Compute(ComputeTask, ConnectionId),
    Init(SessionId),
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub enum ComputeTask {
    RegisterOperation(Box<OperationIr>),
    RegisterTensor(TensorId, TensorData),
    RegisterOrphan(TensorId),
    ReadTensor(TensorIr),
    SyncBackend,
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub struct TaskResponse {
    pub content: TaskResponseContent,
    pub id: ConnectionId,
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub enum TaskResponseContent {
    ReadTensor(TensorData),
    SyncBackend,
}
