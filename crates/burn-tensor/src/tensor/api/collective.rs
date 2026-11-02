use alloc::vec::Vec;

use burn_std::distributed::ReduceOperation;

use crate::{
    AsIndex, Device, Tensor, check,
    check::{TensorCheck, unwrap_dim_index},
    kind::{Basic, Numeric},
    ops::BridgeTensor,
};

impl<const D: usize, K: Numeric> Tensor<D, K> {
    /// Reduces every member's tensor with `op`, and gives every member the result.
    ///
    /// Member `i` is position `i` of `tensors`, each on its own member's device and every one of
    /// one shape and dtype. The outputs are in member order, each on its member's device. A device
    /// may appear twice, which runs the collective over transfers rather than natively.
    ///
    /// A [`ReduceOperation::Mean`] over integers divides by the member count, truncating toward
    /// zero.
    ///
    /// Under autodiff the outputs share one backward step: combine their losses and call `backward`
    /// once.
    ///
    /// # Panics
    ///
    /// - If `tensors` is empty.
    /// - If the members don't all have the same shape and dtype.
    /// - If the members are quantized.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, distributed::ReduceOperation};
    ///
    /// let device = Default::default();
    /// let members = vec![
    ///     Tensor::<1>::from_data([1.0, 2.0], &device),
    ///     Tensor::<1>::from_data([3.0, 4.0], &device),
    /// ];
    ///
    /// let reduced = Tensor::all_reduce(members, ReduceOperation::Sum);
    ///
    /// assert_eq!(reduced[0].clone().try_into_vec_as::<f32>().unwrap(), vec![4.0, 6.0]);
    /// assert_eq!(reduced[1].clone().try_into_vec_as::<f32>().unwrap(), vec![4.0, 6.0]);
    /// ```
    pub fn all_reduce(tensors: Vec<Self>, op: ReduceOperation) -> Vec<Self> {
        check!(TensorCheck::collective("All Reduce", &tensors));

        from_primitives(K::all_reduce(into_primitives(tensors), op))
    }

    /// Reduces every member's tensor with `op`, and gives member `i` chunk `i` along `dim` of the
    /// result.
    ///
    /// Member `i` is position `i` of `tensors`, each on its own member's device and every one of
    /// one shape and dtype. Chunks are as even as possible, with the first members holding one
    /// more. The dimension supports negative indexing.
    ///
    /// A [`ReduceOperation::Mean`] over integers divides by the member count, truncating toward
    /// zero.
    ///
    /// Under autodiff the outputs share one backward step: combine their losses and call `backward`
    /// once.
    ///
    /// # Panics
    ///
    /// - If `tensors` is empty.
    /// - If the members don't all have the same shape and dtype.
    /// - If the members are quantized.
    /// - If `dim` is outside the rank, or holds fewer elements than there are members.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, distributed::ReduceOperation};
    ///
    /// let device = Default::default();
    /// let members = vec![
    ///     Tensor::<1>::from_data([1.0, 2.0], &device),
    ///     Tensor::<1>::from_data([3.0, 4.0], &device),
    /// ];
    ///
    /// let scattered = Tensor::reduce_scatter(members, 0, ReduceOperation::Sum);
    ///
    /// assert_eq!(scattered[0].clone().try_into_vec_as::<f32>().unwrap(), vec![4.0]);
    /// assert_eq!(scattered[1].clone().try_into_vec_as::<f32>().unwrap(), vec![6.0]);
    /// ```
    pub fn reduce_scatter(tensors: Vec<Self>, dim: impl AsIndex, op: ReduceOperation) -> Vec<Self> {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Reduce Scatter");
        check!(TensorCheck::collective("Reduce Scatter", &tensors));
        check!(TensorCheck::collective_chunks(
            "Reduce Scatter",
            &tensors,
            dim
        ));

        from_primitives(K::reduce_scatter(into_primitives(tensors), dim, op))
    }
}

impl<const D: usize, K: Basic> Tensor<D, K> {
    /// Gives every member every member's tensor, concatenated along `dim` in member order.
    ///
    /// Member `i` is position `i` of `tensors`, each on its own member's device and every one of
    /// one shape and dtype. The outputs are in member order, each on its member's device. The
    /// dimension supports negative indexing.
    ///
    /// Under autodiff the outputs share one backward step: combine their losses and call `backward`
    /// once.
    ///
    /// # Panics
    ///
    /// - If `tensors` is empty.
    /// - If the members don't all have the same shape and dtype.
    /// - If the members are quantized.
    /// - If `dim` is outside the rank.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::Tensor;
    ///
    /// let device = Default::default();
    /// let members = vec![
    ///     Tensor::<1>::from_data([1.0, 2.0], &device),
    ///     Tensor::<1>::from_data([3.0, 4.0], &device),
    /// ];
    ///
    /// let gathered = Tensor::all_gather(members, 0);
    ///
    /// assert_eq!(gathered[0].clone().try_into_vec_as::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(gathered[1].clone().try_into_vec_as::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn all_gather(tensors: Vec<Self>, dim: impl AsIndex) -> Vec<Self> {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "All Gather");
        check!(TensorCheck::collective("All Gather", &tensors));

        from_primitives(K::all_gather(into_primitives(tensors), dim))
    }

    /// Gives member `i` chunk `i` along `split_dim` of every member's tensor, concatenated along
    /// `concat_dim` in member order.
    ///
    /// Member `i` is position `i` of `tensors`, each on its own member's device and every one of
    /// one shape and dtype. Chunks are as even as possible, with the first members holding one
    /// more. Both dimensions support negative indexing.
    ///
    /// Under autodiff the outputs share one backward step: combine their losses and call `backward`
    /// once.
    ///
    /// # Panics
    ///
    /// - If `tensors` is empty.
    /// - If the members don't all have the same shape and dtype.
    /// - If the members are quantized.
    /// - If either dimension is outside the rank, or `split_dim` holds fewer elements than there
    ///   are members.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::Tensor;
    ///
    /// let device = Default::default();
    /// let members = vec![
    ///     Tensor::<1>::from_data([1.0, 2.0], &device),
    ///     Tensor::<1>::from_data([3.0, 4.0], &device),
    /// ];
    ///
    /// let exchanged = Tensor::all_to_all(members, 0, 0);
    ///
    /// assert_eq!(exchanged[0].clone().try_into_vec_as::<f32>().unwrap(), vec![1.0, 3.0]);
    /// assert_eq!(exchanged[1].clone().try_into_vec_as::<f32>().unwrap(), vec![2.0, 4.0]);
    /// ```
    pub fn all_to_all(
        tensors: Vec<Self>,
        split_dim: impl AsIndex,
        concat_dim: impl AsIndex,
    ) -> Vec<Self> {
        let split_dim = unwrap_dim_index(split_dim.try_dim_index(D), "All to All");
        let concat_dim = unwrap_dim_index(concat_dim.try_dim_index(D), "All to All");
        check!(TensorCheck::collective("All to All", &tensors));
        check!(TensorCheck::collective_chunks(
            "All to All",
            &tensors,
            split_dim
        ));

        from_primitives(K::all_to_all(
            into_primitives(tensors),
            split_dim,
            concat_dim,
        ))
    }

    /// Gives each of `devices` the tensor, in the order of `devices`.
    ///
    /// To repeat a tensor over a shape rather than over devices, see [`Tensor::expand`].
    ///
    /// # Panics
    ///
    /// - If `devices` is empty.
    /// - If the tensor is quantized.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Device, Tensor};
    ///
    /// let device = Device::default();
    /// let tensor = Tensor::<1>::from_data([1.0, 2.0], &device);
    ///
    /// let members = Tensor::broadcast(tensor, &[device.clone(), device]);
    ///
    /// assert_eq!(members[0].clone().try_into_vec_as::<f32>().unwrap(), vec![1.0, 2.0]);
    /// assert_eq!(members[1].clone().try_into_vec_as::<f32>().unwrap(), vec![1.0, 2.0]);
    /// ```
    pub fn broadcast(tensor: Self, devices: &[Device]) -> Vec<Self> {
        check!(TensorCheck::collective(
            "Broadcast",
            core::slice::from_ref(&tensor)
        ));
        check!(TensorCheck::collective_devices("Broadcast", devices));

        from_primitives(K::broadcast(tensor.primitive, devices))
    }
}

fn into_primitives<const D: usize, K: Basic>(tensors: Vec<Tensor<D, K>>) -> Vec<BridgeTensor> {
    tensors.into_iter().map(|tensor| tensor.primitive).collect()
}

fn from_primitives<const D: usize, K: Basic>(primitives: Vec<BridgeTensor>) -> Vec<Tensor<D, K>> {
    primitives.into_iter().map(Tensor::new).collect()
}
