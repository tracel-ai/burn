use burn_tensor::{
    repr::{OperationDescription, TensorDescription, TensorId},
    DType, TensorData,
};
use serde::{Deserialize, Serialize};

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterOperation {
    pub op: OperationDescription,
    pub position: u64,
    pub client_id: u64,
}
#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub struct ReadTensor {
    pub tensor: TensorDescription,
    pub position: u64,
    pub client_id: u64,
}
#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterTensor {
    pub data: TensorData,
    pub position: u64,
    pub client_id: u64,
}
#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterTensorEmpty {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub position: u64,
    pub client_id: u64,
}
#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterOrphan {
    pub id: TensorId,
    pub position: u64,
    pub client_id: u64,
}
#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub struct SyncBackend {
    pub position: u64,
    pub client_id: u64,
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub struct CloseConnection {
    pub position: u64,
    pub client_id: u64,
}
