use crate::backend::SparseTensor;
use burn_tensor::{
    backend::Backend,
    ops::{BoolTensor, FloatElem, FloatTensor, IntTensor},
    Device, Shape, TensorData,
};
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
        lhs: Self::FloatTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
        sparse: Self::SparseTensorPrimitive<D>,
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

    /// Adds two sparse tensors together.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of adding the two tensors together.
    fn sparse_add<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D>;

    /// Adds a sparse and dense tensor together.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of adding the two tensors together.
    fn sparse_add_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D>;

    /// Adds a scalar to a tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of adding the scalar to the tensor.
    fn sparse_add_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D>;

    /// Subtracts two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of subtracting the two tensors.
    fn sparse_sub<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D>;

    /// Subtracts a dense from a sparse tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor (sparse).
    /// * `rhs` - The right hand side tensor (dense).
    ///
    /// # Returns
    ///
    /// The result of subtracting the two tensors.
    fn sparse_sub_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D>;

    /// Subtracts a scalar from a tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of subtracting the scalar from the tensor.
    fn sparse_sub_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D>;

    /// Multiplies two sparse tensors together.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of multiplying the two tensors together.
    fn sparse_mul<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D>;

    /// Multiplies  a sparse and dense tensor together.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of multiplying the two tensors together.
    fn sparse_mul_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D>;

    /// Multiplies a scalar to a tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of multiplying the scalar with the tensor.
    fn sparse_mul_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D>;

    /// Divides two sparse tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of dividing the two tensors.
    fn sparse_div<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D>;

    /// Divides a sparse and dense tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of dividing the two tensors.
    fn sparse_div_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D>;

    /// Divides a tensor by a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of dividing the tensor by the scalar.
    fn sparse_div_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D>;

    fn sparse_max<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1>;

    fn sparse_max_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> SparseTensor<Self, D>;

    fn sparse_min<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1>;

    fn sparse_min_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> SparseTensor<Self, D>;

    fn sparse_greater<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> BoolTensor<Self, D>;

    fn sparse_greater_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D>;

    fn sparse_greater_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> BoolTensor<Self, D>;

    fn sparse_greater_equal_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D>;

    fn sparse_lower<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> BoolTensor<Self, D>;

    fn sparse_lower_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D>;

    fn sparse_lower_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> BoolTensor<Self, D>;

    fn sparse_lower_equal_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D>;

    fn sparse_abs<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D>;
    fn sparse_sign<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D>;

    fn sparse_powf<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D>;

    fn sparse_powi<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D>;

    fn sparse_powf_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D>;

    fn sparse_powi_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D>;

    fn sparse_clamp<const D: usize>(
        tensor: SparseTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> SparseTensor<Self, D>;

    fn sparse_clamp_min<const D: usize>(
        tensor: SparseTensor<Self, D>,
        min: FloatElem<Self>,
    ) -> SparseTensor<Self, D>;

    fn sparse_clamp_max<const D: usize>(
        tensor: SparseTensor<Self, D>,
        max: FloatElem<Self>,
    ) -> SparseTensor<Self, D>;

    fn sparse_select<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> SparseTensor<Self, D>;

    fn sparse_select_assign<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        values: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D>;

    fn sparse_gather<const D: usize>(
        dim: usize,
        tensor: SparseTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> SparseTensor<Self, D>;

    fn sparse_scatter<const D: usize>(
        dim: usize,
        tensor: SparseTensor<Self, D>,
        indices: IntTensor<Self, D>,
        values: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D>;

    fn sparse_sum<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1>;

    fn sparse_sum_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> SparseTensor<Self, D>;

    fn sparse_prod<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1>;

    fn sparse_prod_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> SparseTensor<Self, D>;

    fn sparse_mean<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1>;

    fn sparse_mean_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> SparseTensor<Self, D>;

    fn sparse_equal_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D>;

    fn sparse_not_equal_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D>;

    fn sparse_remainder_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D>;

    fn sparse_neg<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D>;
}
