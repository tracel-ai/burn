use burn_backend::{
    AllReduceStrategy, ReduceOperation,
    ops::CommunicationTensorOps,
    tensor::{CommunicationTensor, Device, FloatTensor},
};

use crate::{Fusion, FusionBackend};

impl<B: FusionBackend> CommunicationTensorOps<Self> for Fusion<B> {
    fn all_reduce_inplace(
        tensors: Vec<CommunicationTensor<Self>>,
        strategy: AllReduceStrategy,
        op: ReduceOperation,
    ) {
        todo!()
    }

    fn all_broadcast_inplace(
        src_tensor: FloatTensor<Self>,
        dest_tensors: Vec<CommunicationTensor<Self>>,
    ) {
        todo!()
    }

    fn comm_device(tensor: &CommunicationTensor<Self>) -> Device<Self> {
        todo!()
    }

    fn float_data_from_comm(tensor: &CommunicationTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }
}
