use burn_tensor::{
    repr::{
        BaseOperationDescription, OperationDescription, TensorDescription, TensorId, TensorStatus,
    },
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

impl TaskContent {
    pub fn tensors(&self) -> Vec<(TensorId, TensorStatus)> {
        fn from_descriptions(desc: &[&TensorDescription]) -> Vec<(TensorId, TensorStatus)> {
            desc.iter().map(|t| (t.id, t.status.clone())).collect()
        }

        match self {
            TaskContent::RegisterOperation(op) => from_descriptions(&op.nodes()),
            TaskContent::RegisterTensor(tensor_id, _tensor_data) => {
                vec![(*tensor_id, TensorStatus::NotInit)]
            }
            TaskContent::RegisterOrphan(tensor_id) => {
                vec![(*tensor_id, TensorStatus::ReadWrite)]
            }
            TaskContent::ReadTensor(tensor_description) => from_descriptions(&[tensor_description]),
            TaskContent::SyncBackend => vec![],
            TaskContent::FlushBackend => vec![],
        }
    }
}
