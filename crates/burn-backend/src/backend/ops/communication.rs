use crate::{
    AllReduceStrategy, Backend, ReduceOperation,
    tensor::{CommunicationTensor, Device, FloatTensor},
};

/// Operations on communication tensors.
pub trait CommunicationTensorOps<B: Backend> {
    /// Performs an all_reduce operation on the given tensors and replaces the values in-place.
    ///
    /// # Arguments
    ///
    /// * `tensors` - The tensors on which to perform all_reduce.
    /// * `strategy` - The [`AllReduceStrategy`].
    /// * `op` - The [`ReduceOperation`].
    fn all_reduce_inplace(
        tensors: Vec<CommunicationTensor<B>>,
        strategy: AllReduceStrategy,
        op: ReduceOperation,
    );
    /// Performs a broadcast of the given source tensor to the destinations, in-place.
    ///
    /// # Arguments
    ///
    /// * `src_tensors` - A float tensor of the data to broadcast.
    /// * `dest_tensors` - The tensors on which to perform the broadcast in-place.
    fn all_broadcast_inplace(src_tensor: FloatTensor<B>, dest_tensors: Vec<CommunicationTensor<B>>);
    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn comm_device(tensor: &CommunicationTensor<B>) -> Device<B>;
    /// Creates a float tensor from the current data in the communication tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing a copy of the data of the given tensor.
    fn float_data_from_comm(tensor: &CommunicationTensor<B>) -> FloatTensor<B>;
}
