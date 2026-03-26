use burn_backend::{
    ReduceOperation,
    ops::{CommunicationTensorOps, TensorRef},
};

use crate::{Fusion, FusionBackend};

impl<B: FusionBackend> CommunicationTensorOps<Self> for Fusion<B> {
    unsafe fn all_reduce_in_place(tensors: Vec<TensorRef<Self>>, op: ReduceOperation) {
        println!("Fusion all_reduce");
        let tensors = tensors
            .iter()
            .map(|t| {
                let t = unsafe { (*t.0).clone() };
                let client = t.client.clone();
                let mut t = client.resolve_tensor_float::<B>(t);
                TensorRef(&mut t)
            })
            .collect();
        unsafe { B::all_reduce_in_place(tensors, op) };
    }
}
