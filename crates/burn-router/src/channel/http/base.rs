use burn_tensor::{
    repr::{OperationDescription, TensorDescription, TensorId},
    DType, TensorData,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterOperation {
    /// TODO:
    pub op: OperationDescription,
    pub index: u64,
}
/// TODO:
#[derive(Serialize, Deserialize, Debug)]
pub struct ReadTensor {
    /// TODO:
    pub tensor: TensorDescription,
    pub index: u64,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterTensor {
    /// TODO:
    pub data: TensorData,
    pub index: u64,
}
/// TODO:
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterTensorEmpty {
    /// TODO:
    pub shape: Vec<usize>,

    /// TODO:
    pub dtype: DType,
    pub index: u64,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterOrphan {
    /// TODO:
    pub id: TensorId,
    pub index: u64,
}
/// TODO:
#[derive(Serialize, Deserialize, Debug)]
pub struct SyncBackend {
    pub index: u64,
}
