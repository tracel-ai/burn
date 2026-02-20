use burn_backend::{
    AllReduceStrategy, Backend, ReduceOperation,
    ops::CommunicationTensorOps,
    tensor::{CommunicationTensor, Device, FloatTensor},
};

use crate::{Candle, FloatCandleElement, IntCandleElement};

impl<F: FloatCandleElement, I: IntCandleElement> CommunicationTensorOps<Self> for Candle<F, I> {
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
