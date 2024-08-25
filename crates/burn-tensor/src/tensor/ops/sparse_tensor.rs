use super::{BoolTensor, FloatElem, FloatTensor, IntTensor, QuantizedTensor};
use crate::TensorRepr;
use crate::{
    backend::Backend, Bool, Device, Float, Int, ReprPrimitive, Shape, Sparse, SparseStorage,
    TensorData, TensorKind,
};
use core::{future::Future, ops::Range};

pub trait SparseTensorOps<SR: SparseStorage<B>, B: Backend>:
    SparseFloatOps<SR, B> + SparseBoolOps<SR, B> + SparseIntOps<SR, B>
{
}

pub trait SparseFloatOps<SR: SparseStorage<B>, B: Backend>
where
    (B, Float, Sparse<B, SR>): TensorRepr,
    (B, Bool, Sparse<B, SR>): TensorRepr,
{
    fn float_to_sparse<const D: usize>(
        dense: B::FloatTensorPrimitive<D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_empty<const D: usize>(
        shape: Shape<D>,
        device: &Device<B>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_to_dense<const D: usize>(
        sparse: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> B::FloatTensorPrimitive<D>;

    fn float_spmm<const D: usize>(
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: <Float as TensorKind<B>>::Primitive<D>,
    ) -> B::FloatTensorPrimitive<D>;

    fn float_sddmm<const D: usize>(
        lhs: B::FloatTensorPrimitive<D>,
        rhs: B::FloatTensorPrimitive<D>,
        sparse: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_coalesce_sum<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_remove_zeros<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_number_nonzero<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> usize;

    fn float_density<const D: usize>(sparse: ReprPrimitive<B, Float, Sparse<B, SR>, D>) -> f32;

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
        tensor: SR::SparsePrimitive<Float, D1>,
        indices: [Range<usize>; D2],
    ) -> SR::SparsePrimitive<Float, D1>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn float_device<const D: usize>(
        tensor: &ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> Device<B>;

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
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        device: &Device<B>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    /// Gets the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn float_shape<const D: usize>(tensor: &ReprPrimitive<B, Float, Sparse<B, SR>, D>) -> Shape<D>;

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: SR::SparsePrimitive<Float, D1>,
        shape: Shape<D2>,
    ) -> SR::SparsePrimitive<Float, D2>;

    fn float_transpose<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_swap_dims<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        dim1: usize,
        dim2: usize,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_permute<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        axes: &[usize],
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_flip<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        axes: &[usize],
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: SR::SparsePrimitive<Float, D1>,
        ranges: [Range<usize>; D2],
        value: SR::SparsePrimitive<Float, D1>,
    ) -> SR::SparsePrimitive<Float, D1>;

    fn float_repeat_dim<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        dim: usize,
        times: usize,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_cat<const D: usize>(
        tensors: Vec<ReprPrimitive<B, Float, Sparse<B, SR>, D>>,
        dim: usize,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_equal<const D: usize>(
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn float_not_equal<const D: usize>(
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn float_any<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> SR::SparsePrimitive<Bool, 1>;

    fn float_any_dim<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        dim: usize,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn float_all<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> SR::SparsePrimitive<Bool, 1>;

    fn float_all_dim<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        dim: usize,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: SR::SparsePrimitive<Float, D1>,
        shape: Shape<D2>,
    ) -> SR::SparsePrimitive<Float, D2>;

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
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

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
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

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
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

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
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: FloatElem<B>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

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
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

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
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: FloatElem<B>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_max<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_max_dim<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        dim: usize,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_min<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_min_dim<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        dim: usize,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_abs<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;
    fn float_sign<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_powf<const D: usize>(
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_powi<const D: usize>(
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_powf_scalar<const D: usize>(
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: FloatElem<B>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_powi_scalar<const D: usize>(
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: FloatElem<B>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_clamp<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        min: FloatElem<B>,
        max: FloatElem<B>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_clamp_min<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        min: FloatElem<B>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_clamp_max<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        max: FloatElem<B>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_select<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_select_assign<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
        values: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        indices: IntTensor<B, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        indices: IntTensor<B, D>,
        values: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_sum<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> SR::SparsePrimitive<Float, 1>;

    fn float_sum_dim<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        dim: usize,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_prod_dim<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        dim: usize,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_mean<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> SR::SparsePrimitive<Float, 1>;

    fn float_mean_dim<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        dim: usize,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_remainder_scalar<const D: usize>(
        lhs: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
        rhs: FloatElem<B>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;

    fn float_neg<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D>;
}

pub trait SparseBoolOps<SR: SparseStorage<B>, B: Backend> {
    fn bool_to_sparse<const D: usize>(
        dense: B::BoolTensorPrimitive<D>,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_empty<const D: usize>(
        shape: Shape<D>,
        device: &Device<B>,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_shape<const D: usize>(tensor: &SR::SparsePrimitive<Bool, D>) -> Shape<D>;

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: SR::SparsePrimitive<Bool, D1>,
        shape: Shape<D2>,
    ) -> SR::SparsePrimitive<Bool, D2>;

    fn bool_transpose<const D: usize>(
        tensor: SR::SparsePrimitive<Bool, D>,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_swap_dims<const D: usize>(
        tensor: SR::SparsePrimitive<Bool, D>,
        dim1: usize,
        dim2: usize,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_permute<const D: usize>(
        tensor: SR::SparsePrimitive<Bool, D>,
        axes: &[usize],
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_flip<const D: usize>(
        tensor: SR::SparsePrimitive<Bool, D>,
        axes: &[usize],
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: SR::SparsePrimitive<Bool, D1>,
        indices: [Range<usize>; D2],
    ) -> SR::SparsePrimitive<Bool, D1>;

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: SR::SparsePrimitive<Bool, D1>,
        ranges: [Range<usize>; D2],
        value: SR::SparsePrimitive<Bool, D1>,
    ) -> SR::SparsePrimitive<Bool, D1>;

    fn bool_device<const D: usize>(tensor: &SR::SparsePrimitive<Bool, D>) -> Device<B>;

    fn bool_to_device<const D: usize>(
        tensor: SR::SparsePrimitive<Bool, D>,
        device: &Device<B>,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_repeat_dim<const D: usize>(
        tensor: SR::SparsePrimitive<Bool, D>,
        dim: usize,
        times: usize,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_cat<const D: usize>(
        tensors: Vec<SR::SparsePrimitive<Bool, D>>,
        dim: usize,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_equal<const D: usize>(
        lhs: SR::SparsePrimitive<Bool, D>,
        rhs: SR::SparsePrimitive<Bool, D>,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_not_equal<const D: usize>(
        lhs: SR::SparsePrimitive<Bool, D>,
        rhs: SR::SparsePrimitive<Bool, D>,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_any<const D: usize>(
        tensor: SR::SparsePrimitive<Bool, D>,
    ) -> SR::SparsePrimitive<Bool, 1>;

    fn bool_any_dim<const D: usize>(
        tensor: SR::SparsePrimitive<Bool, D>,
        dim: usize,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_all<const D: usize>(
        tensor: SR::SparsePrimitive<Bool, D>,
    ) -> SR::SparsePrimitive<Bool, 1>;

    fn bool_all_dim<const D: usize>(
        tensor: SR::SparsePrimitive<Bool, D>,
        dim: usize,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn bool_expand<const D1: usize, const D2: usize>(
        tensor: SR::SparsePrimitive<Bool, D1>,
        shape: Shape<D2>,
    ) -> SR::SparsePrimitive<Bool, D2>;
}

pub trait SparseIntOps<SR: SparseStorage<B>, B: Backend> {
    fn int_empty<const D: usize>(
        shape: Shape<D>,
        device: &Device<B>,
    ) -> SR::SparsePrimitive<Int, D>;

    fn int_shape<const D: usize>(tensor: &SR::SparsePrimitive<Int, D>) -> Shape<D>;

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: SR::SparsePrimitive<Int, D1>,
        shape: Shape<D2>,
    ) -> SR::SparsePrimitive<Int, D2>;

    fn int_transpose<const D: usize>(
        tensor: SR::SparsePrimitive<Int, D>,
    ) -> SR::SparsePrimitive<Int, D>;

    fn int_swap_dims<const D: usize>(
        tensor: SR::SparsePrimitive<Int, D>,
        dim1: usize,
        dim2: usize,
    ) -> SR::SparsePrimitive<Int, D>;

    fn int_permute<const D: usize>(
        tensor: SR::SparsePrimitive<Int, D>,
        axes: &[usize],
    ) -> SR::SparsePrimitive<Int, D>;

    fn int_flip<const D: usize>(
        tensor: SR::SparsePrimitive<Int, D>,
        axes: &[usize],
    ) -> SR::SparsePrimitive<Int, D>;

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: SR::SparsePrimitive<Int, D1>,
        indices: [Range<usize>; D2],
    ) -> SR::SparsePrimitive<Int, D1>;

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: SR::SparsePrimitive<Int, D1>,
        ranges: [Range<usize>; D2],
        value: SR::SparsePrimitive<Int, D1>,
    ) -> SR::SparsePrimitive<Int, D1>;

    fn int_device<const D: usize>(tensor: &SR::SparsePrimitive<Int, D>) -> Device<B>;

    fn int_to_device<const D: usize>(
        tensor: SR::SparsePrimitive<Int, D>,
        device: &Device<B>,
    ) -> SR::SparsePrimitive<Int, D>;

    fn int_repeat_dim<const D: usize>(
        tensor: SR::SparsePrimitive<Int, D>,
        dim: usize,
        times: usize,
    ) -> SR::SparsePrimitive<Int, D>;

    fn int_cat<const D: usize>(
        tensors: Vec<SR::SparsePrimitive<Int, D>>,
        dim: usize,
    ) -> SR::SparsePrimitive<Int, D>;

    fn int_equal<const D: usize>(
        lhs: SR::SparsePrimitive<Int, D>,
        rhs: SR::SparsePrimitive<Int, D>,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn int_not_equal<const D: usize>(
        lhs: SR::SparsePrimitive<Int, D>,
        rhs: SR::SparsePrimitive<Int, D>,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn int_any<const D: usize>(tensor: SR::SparsePrimitive<Int, D>)
        -> SR::SparsePrimitive<Bool, 1>;

    fn int_any_dim<const D: usize>(
        tensor: SR::SparsePrimitive<Int, D>,
        dim: usize,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn int_all<const D: usize>(tensor: SR::SparsePrimitive<Int, D>)
        -> SR::SparsePrimitive<Bool, 1>;

    fn int_all_dim<const D: usize>(
        tensor: SR::SparsePrimitive<Int, D>,
        dim: usize,
    ) -> SR::SparsePrimitive<Bool, D>;

    fn int_expand<const D1: usize, const D2: usize>(
        tensor: SR::SparsePrimitive<Int, D1>,
        shape: Shape<D2>,
    ) -> SR::SparsePrimitive<Int, D2>;
}
