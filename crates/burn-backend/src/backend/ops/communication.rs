use std::sync::Arc;

use crate::{
    AllReduceStrategy, Backend, ReduceOperation,
    tensor::{Device, FloatTensor},
};

pub(crate) unsafe fn reduce_sum_centralized<B: Backend>(
    tensors: &Vec<TensorRef<B>>,
    central_device: &B::Device,
) -> B::FloatTensorPrimitive {
    let mut central_tensor = (**tensors[0].0).clone();
    for tensor in tensors {
        let rhs = B::float_to_device((**tensor.0).clone(), &central_device);
        central_tensor = B::float_add(central_tensor, rhs);
    }

    central_tensor
}

pub(crate) unsafe fn all_reduce_inplace_sum_centralized<B: Backend>(
    tensors: Vec<TensorRef<B>>,
    op: ReduceOperation,
) {
    let devices: Vec<B::Device> = tensors
        .iter()
        .map(|tensor| B::comm_device(tensor))
        .collect();
    let central_device = devices.get(0).unwrap();

    // TODO: inplace?
    let mut central_tensor = reduce_sum_centralized::<B>(&tensors, &central_device);

    if op == ReduceOperation::Mean {
        // Apply mean division
        let div = (tensors.len() as f32).into();
        central_tensor = B::float_div_scalar(central_tensor, div);
    }

    // Broadcast result to all
    B::all_broadcast_inplace(central_tensor, tensors);
}

#[derive(Clone)]
pub struct TensorRef<B: Backend>(pub Arc<*mut FloatTensor<B>>);
unsafe impl<B> Sync for TensorRef<B> where B: Backend {}
unsafe impl<B> Send for TensorRef<B> where B: Backend {}

/// Operations on communication tensors.
pub trait CommunicationTensorOps<B: Backend> {
    /// Performs an all_reduce operation on the given tensors and replaces the values in-place.
    ///
    /// # Arguments
    ///
    /// * `tensors` - The tensors on which to perform all_reduce.
    /// * `strategy` - The [`AllReduceStrategy`].
    /// * `op` - The [`ReduceOperation`].
    unsafe fn all_reduce_inplace(
        tensors: Vec<TensorRef<B>>,
        strategy: AllReduceStrategy,
        op: ReduceOperation,
    ) {
        match strategy {
            AllReduceStrategy::Centralized => all_reduce_inplace_sum_centralized::<B>(tensors, op),
            // AllReduceStrategy::Tree(arity) => all_reduce_sum_tree::<B>(tensors, *arity),
            // AllReduceStrategy::Ring => all_reduce_sum_ring::<B>(tensors),
            AllReduceStrategy::Tree(arity) => todo!(),
            AllReduceStrategy::Ring => todo!(),
        };
    }
    /// Performs a broadcast of the given source tensor to the destinations, in-place.
    ///
    /// # Arguments
    ///
    /// * `src_tensors` - A float tensor of the data to broadcast.
    /// * `dest_tensors` - The tensors on which to perform the broadcast in-place.
    unsafe fn all_broadcast_inplace(src_tensor: FloatTensor<B>, dest_tensors: Vec<TensorRef<B>>) {
        for dest in dest_tensors {
            let device = B::comm_device(&dest);
            let tensor_float = B::float_to_device(src_tensor.clone(), &device);
            (**dest.0) = tensor_float;
        }
    }
    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn comm_device(tensor: &TensorRef<B>) -> Device<B> {
        unsafe { B::float_device(&(**tensor.0)) }
    }
    /// Creates a float tensor from the current data in the communication tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing a copy of the data of the given tensor.
    fn float_data_from_comm(tensor: &TensorRef<B>) -> FloatTensor<B> {
        unsafe { (**tensor.0).clone() }
    }
}
