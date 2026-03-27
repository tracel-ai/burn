use burn_backend::{
    ReduceOperation,
    ops::{CommunicationTensorOps, TensorRef},
};

use crate::{Fusion, FusionBackend};

// TODO : Fusion won't work with DDP right now.
impl<B: FusionBackend> CommunicationTensorOps<Self> for Fusion<B> {
    unsafe fn all_reduce_in_place(_tensors: Vec<TensorRef<Self>>, _op: ReduceOperation) {
        todo!()
    }
}
