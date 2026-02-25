use burn_backend::{
    AllReduceStrategy, Backend, ReduceOperation,
    ops::{CommunicationTensorOps, TensorRef},
    tensor::{CommunicationTensor, Device, FloatTensor},
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor::AutodiffTensor};

impl<B: Backend, C: CheckpointStrategy> CommunicationTensorOps<Self> for Autodiff<B, C> {
    // unsafe fn all_reduce_inplace(
    //     tensors: Vec<TensorRef<Self>>,
    //     strategy: AllReduceStrategy,
    //     op: ReduceOperation,
    // ) {
    //     B::all_reduce_inplace(tensor, strategy, op);
    // }

    // fn all_broadcast_inplace(
    //     src_tensor: FloatTensor<Self>,
    //     dest_tensors: Vec<CommunicationTensor<Self>>,
    // ) {
    //     B::all_broadcast_inplace(src_tensor.primitive, dest_tensors);
    // }

    // fn comm_device(tensor: &CommunicationTensor<Self>) -> Device<Self> {
    //     B::comm_device(tensor)
    // }

    // fn float_data_from_comm(
    //     tensor: &burn_backend::tensor::CommunicationTensor<B>,
    // ) -> FloatTensor<Self> {
    //     AutodiffTensor::new(B::float_data_from_comm(tensor))
    // }
}
