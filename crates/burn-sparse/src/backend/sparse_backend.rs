use crate::backend::SparseTensor;
use burn_tensor::{backend::Backend, ops::BoolTensor, Device, Shape, TensorData};
use core::{future::Future, ops::Range};

pub trait SparseBackend: Backend {
    type SparseTensorPrimitive<const D: usize>: Clone + Send + 'static + core::fmt::Debug;

    fn sparse_empty<const D: usize>(
        shape: Shape<D>,
        device: &Device<Self>,
    ) -> SparseTensor<Self, D>;

    fn sparse_to_sparse<const D: usize>(
        dense: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D>;

    fn sparse_to_dense<const D: usize>(
        sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D>;

    fn sparse_spmm<const D: usize>(
        lhs: Self::SparseTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D>;

    fn sparse_sddmm<const D: usize>(
        lhs: Self::SparseTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D>;

    fn sparse_coalesce_sum<const D: usize>(
        tensor: Self::SparseTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D>;

    fn sparse_nonzero<const D: usize>(tensor: Self::SparseTensorPrimitive<D>) -> usize;

    fn sparse_density<const D: usize>(sparse: Self::SparseTensorPrimitive<D>) -> f32;

    /// Gets the element at the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `indices` - The indices.
    ///
    /// # Returns
    ///
    /// The elements at the given indices.
    fn sparse_slice<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        indices: [Range<usize>; D2],
    ) -> SparseTensor<Self, D1>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn sparse_device<const D: usize>(tensor: &SparseTensor<Self, D>) -> Device<Self>;

    /// Moves the tensor to the given device.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `device` - The device to move the tensor to.
    ///
    /// # Returns
    ///
    /// The tensor on the given device.
    fn sparse_to_device<const D: usize>(
        tensor: SparseTensor<Self, D>,
        device: &Device<Self>,
    ) -> SparseTensor<Self, D>;

    /// Gets the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn sparse_shape<const D: usize>(tensor: &SparseTensor<Self, D>) -> Shape<D>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn sparse_into_data<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> impl Future<Output = TensorData> + Send;

    /// Creates a tensor from the data structure.
    ///
    /// # Arguments
    ///
    /// * `data` - The data structure.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the data.
    fn sparse_from_data<const D: usize>(
        data: TensorData,
        device: &Device<Self>,
    ) -> SparseTensor<Self, D>;

    fn sparse_reshape<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> SparseTensor<Self, D2>;

    fn sparse_transpose<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D>;

    fn sparse_swap_dims<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> SparseTensor<Self, D>;

    fn sparse_permute<const D: usize>(
        tensor: SparseTensor<Self, D>,
        axes: &[usize],
    ) -> SparseTensor<Self, D>;

    fn sparse_flip<const D: usize>(
        tensor: SparseTensor<Self, D>,
        axes: &[usize],
    ) -> SparseTensor<Self, D>;

    fn sparse_slice_assign<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: SparseTensor<Self, D1>,
    ) -> SparseTensor<Self, D1>;

    fn sparse_repeat<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> SparseTensor<Self, D>;

    fn sparse_cat<const D: usize>(
        tensors: Vec<SparseTensor<Self, D>>,
        dim: usize,
    ) -> SparseTensor<Self, D>;

    fn sparse_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> BoolTensor<Self, D>;

    fn sparse_not_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> BoolTensor<Self, D>;

    fn sparse_any<const D: usize>(tensor: SparseTensor<Self, D>) -> BoolTensor<Self, 1>;

    fn sparse_any_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> BoolTensor<Self, D>;

    fn sparse_all<const D: usize>(tensor: SparseTensor<Self, D>) -> BoolTensor<Self, 1>;

    fn sparse_all_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> BoolTensor<Self, D>;

    fn sparse_expand<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> SparseTensor<Self, D2>;
}
