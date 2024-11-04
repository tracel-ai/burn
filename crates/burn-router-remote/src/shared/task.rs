use burn_tensor::{
    repr::{OperationDescription, TensorDescription, TensorId},
    TensorData,
};
use serde::{Deserialize, Serialize};

#[allow(missing_docs)]
#[derive(new, Serialize, Deserialize, Debug, Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub struct ConnectionId {
    pub position: u64,
    pub stream_id: u64,
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub struct Task {
    pub content: TaskContent,
    pub id: ConnectionId,
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub enum TaskContent {
    RegisterOperation(OperationDescription),
    RegisterTensor(TensorId, TensorData),
    RegisterOrphan(TensorId),
    ReadTensor(TensorDescription),
    SyncBackend,
    FlushBackend,
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
    FlushBackend,
}
