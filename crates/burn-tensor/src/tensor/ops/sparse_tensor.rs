use super::{BoolTensor, FloatElem, FloatTensor, IntTensor, QuantizedTensor};
use crate::{backend::Backend, Device, Float, Shape, SparseRepr, TensorData, TensorKind};
use core::{future::Future, ops::Range};

pub trait SparseTensorOps<R: SparseRepr<B>, B: Backend>: SparseFloatOps<R, B> {}

pub trait SparseFloatOps<R: SparseRepr<B>, B: Backend> {
    fn float_to_sparse<const D: usize>(
        dense: B::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_empty<const D: usize>(
        shape: Shape<D>,
        device: &Device<B>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_to_dense<const D: usize>(
        sparse: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_spmm<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: B::FloatTensorPrimitive<D>,
    ) -> B::FloatTensorPrimitive<D>;

    fn float_sddmm<const D: usize>(
        lhs: B::FloatTensorPrimitive<D>,
        rhs: B::FloatTensorPrimitive<D>,
        sparse: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_coalesce_sum<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_remove_zeros<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_nonzero<const D: usize>(tensor: R::FloatTensorPrimitive<D>) -> usize;

    fn float_density<const D: usize>(sparse: R::FloatTensorPrimitive<D>) -> f32;

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
    fn float_slice<const D1: usize, const D2: usize>(
        tensor: R::FloatTensorPrimitive<D1>,
        indices: [Range<usize>; D2],
    ) -> R::FloatTensorPrimitive<D1>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn float_device<const D: usize>(tensor: &R::FloatTensorPrimitive<D>) -> Device<B>;

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
    fn float_to_device<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        device: &Device<B>,
    ) -> R::FloatTensorPrimitive<D>;

    /// Gets the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn float_shape<const D: usize>(tensor: &R::FloatTensorPrimitive<D>) -> Shape<D>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn float_into_data<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
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
    fn float_from_data<const D: usize>(
        data: TensorData,
        device: &Device<B>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: R::FloatTensorPrimitive<D1>,
        shape: Shape<D2>,
    ) -> R::FloatTensorPrimitive<D2>;

    fn float_transpose<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_swap_dims<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_permute<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        axes: &[usize],
    ) -> R::FloatTensorPrimitive<D>;

    fn float_flip<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        axes: &[usize],
    ) -> R::FloatTensorPrimitive<D>;

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: R::FloatTensorPrimitive<D1>,
        ranges: [Range<usize>; D2],
        value: R::FloatTensorPrimitive<D1>,
    ) -> R::FloatTensorPrimitive<D1>;

    fn float_repeat_dim<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        dim: usize,
        times: usize,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_cat<const D: usize>(
        tensors: Vec<R::FloatTensorPrimitive<D>>,
        dim: usize,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_equal<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_not_equal<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_any<const D: usize>(tensor: R::FloatTensorPrimitive<D>) -> R::BoolTensorPrimitive<1>;

    fn float_any_dim<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        dim: usize,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_all<const D: usize>(tensor: R::FloatTensorPrimitive<D>) -> R::BoolTensorPrimitive<1>;

    fn float_all_dim<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        dim: usize,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: R::FloatTensorPrimitive<D1>,
        shape: Shape<D2>,
    ) -> R::FloatTensorPrimitive<D2>;

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
    fn float_add<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

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
    fn float_add_dense<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

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
    fn float_add_scalar<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::FloatTensorPrimitive<D>;

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
    fn float_sub<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

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
    fn float_sub_dense<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

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
    fn float_sub_scalar<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::FloatTensorPrimitive<D>;

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
    fn float_mul<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

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
    fn float_mul_dense<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

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
    fn float_mul_scalar<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::FloatTensorPrimitive<D>;

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
    fn float_div<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

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
    fn float_div_dense<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

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
    fn float_div_scalar<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_max<const D: usize>(tensor: R::FloatTensorPrimitive<D>) -> R::FloatTensorPrimitive<D>;

    fn float_max_dim<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        dim: usize,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_min<const D: usize>(tensor: R::FloatTensorPrimitive<D>) -> R::FloatTensorPrimitive<D>;

    fn float_min_dim<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        dim: usize,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_greater<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_greater_elem<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_greater_equal<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_greater_equal_elem<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_lower<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_lower_elem<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_lower_equal<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_lower_equal_elem<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_abs<const D: usize>(tensor: R::FloatTensorPrimitive<D>) -> R::FloatTensorPrimitive<D>;
    fn float_sign<const D: usize>(tensor: R::FloatTensorPrimitive<D>)
        -> R::FloatTensorPrimitive<D>;

    fn float_powf<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_powi<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_powf_scalar<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_powi_scalar<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_clamp<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        min: FloatElem<B>,
        max: FloatElem<B>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_clamp_min<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        min: FloatElem<B>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_clamp_max<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        max: FloatElem<B>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_select<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        dim: usize,
        indices: IntTensor<B, 1>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_select_assign<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        dim: usize,
        indices: IntTensor<B, 1>,
        values: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: R::FloatTensorPrimitive<D>,
        indices: IntTensor<B, D>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: R::FloatTensorPrimitive<D>,
        indices: IntTensor<B, D>,
        values: R::FloatTensorPrimitive<D>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_sum<const D: usize>(tensor: R::FloatTensorPrimitive<D>) -> R::FloatTensorPrimitive<D>;

    fn float_sum_dim<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        dim: usize,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_prod<const D: usize>(tensor: R::FloatTensorPrimitive<D>)
        -> R::FloatTensorPrimitive<D>;

    fn float_prod_dim<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        dim: usize,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_mean<const D: usize>(tensor: R::FloatTensorPrimitive<D>)
        -> R::FloatTensorPrimitive<D>;

    fn float_mean_dim<const D: usize>(
        tensor: R::FloatTensorPrimitive<D>,
        dim: usize,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_equal_elem<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_not_equal_elem<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::BoolTensorPrimitive<D>;

    fn float_remainder_scalar<const D: usize>(
        lhs: R::FloatTensorPrimitive<D>,
        rhs: FloatElem<B>,
    ) -> R::FloatTensorPrimitive<D>;

    fn float_neg<const D: usize>(tensor: R::FloatTensorPrimitive<D>) -> R::FloatTensorPrimitive<D>;
}

pub trait SparseIntOps<R: SparseRepr<B>, B: Backend> {}

// pub trait SparseBoolOps<R: SparseRepr<B>, B: Backend> {}
