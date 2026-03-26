use burn_backend::{
    ReduceOperation,
    ops::{CommunicationTensorOps, TensorRef},
};

use crate::{Fusion, FusionBackend};

impl<B: FusionBackend> CommunicationTensorOps<Self> for Fusion<B> {
    unsafe fn all_reduce_in_place(tensors: Vec<TensorRef<Self>>, op: ReduceOperation) {
        let client = unsafe { (*tensors.first().unwrap().0).clone().client.clone() };
        client.all_reduce_in_place::<B>(
            tensors
                .into_iter()
                .map(|t| unsafe { (*t.0).clone() })
                .collect(),
            op,
        );
    }
}
