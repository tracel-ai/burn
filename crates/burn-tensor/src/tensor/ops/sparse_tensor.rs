use super::{BoolTensor, FloatElem, FloatTensor, IntTensor, QuantizedTensor};
use crate::{
    backend::Backend, Bool, Device, Float, Int, Shape, Sparse, SparseRepr, TensorData, TensorKind,
};
use core::{future::Future, ops::Range};

pub trait SparseTensorOps<R: SparseRepr<B>, B: Backend>:
    SparseFloatOps<R, B> + SparseBoolOps<R, B> + SparseIntOps<R, B>
{
}

pub trait SparseFloatOps<R: SparseRepr<B>, B: Backend> {
    fn float_to_sparse<const D: usize>(dense: B::FloatTensorPrimitive<D>)
        -> R::Primitive<Float, D>;

    fn float_empty<const D: usize>(shape: Shape<D>, device: &Device<B>) -> R::Primitive<Float, D>;

    fn float_to_dense<const D: usize>(sparse: R::Primitive<Float, D>)
        -> B::FloatTensorPrimitive<D>;

    fn float_spmm<const D: usize>(
        lhs: R::Primitive<Float, D>,
        rhs: <Float as TensorKind<B, Sparse<R, B>>>::DensePrimitive<D>,
    ) -> B::FloatTensorPrimitive<D>;

    fn float_sddmm<const D: usize>(
        lhs: B::FloatTensorPrimitive<D>,
        rhs: B::FloatTensorPrimitive<D>,
        sparse: R::Primitive<Float, D>,
    ) -> R::Primitive<Float, D>;

    fn float_coalesce_sum<const D: usize>(tensor: R::Primitive<Float, D>)
        -> R::Primitive<Float, D>;

    fn float_remove_zeros<const D: usize>(tensor: R::Primitive<Float, D>)
        -> R::Primitive<Float, D>;

    fn float_number_nonzero<const D: usize>(tensor: R::Primitive<Float, D>) -> usize;

    fn float_density<const D: usize>(sparse: R::Primitive<Float, D>) -> f32;

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
        tensor: R::Primitive<Float, D1>,
        indices: [Range<usize>; D2],
    ) -> R::Primitive<Float, D1>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn float_device<const D: usize>(tensor: &R::Primitive<Float, D>) -> Device<B>;

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
        tensor: R::Primitive<Float, D>,
        device: &Device<B>,
    ) -> R::Primitive<Float, D>;

    /// Gets the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn float_shape<const D: usize>(tensor: &R::Primitive<Float, D>) -> Shape<D>;

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
        tensor: R::Primitive<Float, D>,
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
    ) -> R::Primitive<Float, D>;

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: R::Primitive<Float, D1>,
        shape: Shape<D2>,
    ) -> R::Primitive<Float, D2>;

    fn float_transpose<const D: usize>(tensor: R::Primitive<Float, D>) -> R::Primitive<Float, D>;

    fn float_swap_dims<const D: usize>(
        tensor: R::Primitive<Float, D>,
        dim1: usize,
        dim2: usize,
    ) -> R::Primitive<Float, D>;

    fn float_permute<const D: usize>(
        tensor: R::Primitive<Float, D>,
        axes: &[usize],
    ) -> R::Primitive<Float, D>;

    fn float_flip<const D: usize>(
        tensor: R::Primitive<Float, D>,
        axes: &[usize],
    ) -> R::Primitive<Float, D>;

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: R::Primitive<Float, D1>,
        ranges: [Range<usize>; D2],
        value: R::Primitive<Float, D1>,
    ) -> R::Primitive<Float, D1>;

    fn float_repeat_dim<const D: usize>(
        tensor: R::Primitive<Float, D>,
        dim: usize,
        times: usize,
    ) -> R::Primitive<Float, D>;

    fn float_cat<const D: usize>(
        tensors: Vec<R::Primitive<Float, D>>,
        dim: usize,
    ) -> R::Primitive<Float, D>;

    fn float_equal<const D: usize>(
        lhs: R::Primitive<Float, D>,
        rhs: R::Primitive<Float, D>,
    ) -> R::Primitive<Bool, D>;

    fn float_not_equal<const D: usize>(
        lhs: R::Primitive<Float, D>,
        rhs: R::Primitive<Float, D>,
    ) -> R::Primitive<Bool, D>;

    fn float_any<const D: usize>(tensor: R::Primitive<Float, D>) -> R::Primitive<Bool, 1>;

    fn float_any_dim<const D: usize>(
        tensor: R::Primitive<Float, D>,
        dim: usize,
    ) -> R::Primitive<Bool, D>;

    fn float_all<const D: usize>(tensor: R::Primitive<Float, D>) -> R::Primitive<Bool, 1>;

    fn float_all_dim<const D: usize>(
        tensor: R::Primitive<Float, D>,
        dim: usize,
    ) -> R::Primitive<Bool, D>;

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: R::Primitive<Float, D1>,
        shape: Shape<D2>,
    ) -> R::Primitive<Float, D2>;

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
        lhs: R::Primitive<Float, D>,
        rhs: R::Primitive<Float, D>,
    ) -> R::Primitive<Float, D>;

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
        lhs: R::Primitive<Float, D>,
        rhs: R::Primitive<Float, D>,
    ) -> R::Primitive<Float, D>;

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
        lhs: R::Primitive<Float, D>,
        rhs: R::Primitive<Float, D>,
    ) -> R::Primitive<Float, D>;

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
        lhs: R::Primitive<Float, D>,
        rhs: FloatElem<B>,
    ) -> R::Primitive<Float, D>;

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
        lhs: R::Primitive<Float, D>,
        rhs: R::Primitive<Float, D>,
    ) -> R::Primitive<Float, D>;

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
        lhs: R::Primitive<Float, D>,
        rhs: FloatElem<B>,
    ) -> R::Primitive<Float, D>;

    fn float_max<const D: usize>(tensor: R::Primitive<Float, D>) -> R::Primitive<Float, D>;

    fn float_max_dim<const D: usize>(
        tensor: R::Primitive<Float, D>,
        dim: usize,
    ) -> R::Primitive<Float, D>;

    fn float_min<const D: usize>(tensor: R::Primitive<Float, D>) -> R::Primitive<Float, D>;

    fn float_min_dim<const D: usize>(
        tensor: R::Primitive<Float, D>,
        dim: usize,
    ) -> R::Primitive<Float, D>;

    fn float_abs<const D: usize>(tensor: R::Primitive<Float, D>) -> R::Primitive<Float, D>;
    fn float_sign<const D: usize>(tensor: R::Primitive<Float, D>) -> R::Primitive<Float, D>;

    fn float_powf<const D: usize>(
        lhs: R::Primitive<Float, D>,
        rhs: R::Primitive<Float, D>,
    ) -> R::Primitive<Float, D>;

    fn float_powi<const D: usize>(
        lhs: R::Primitive<Float, D>,
        rhs: R::Primitive<Float, D>,
    ) -> R::Primitive<Float, D>;

    fn float_powf_scalar<const D: usize>(
        lhs: R::Primitive<Float, D>,
        rhs: FloatElem<B>,
    ) -> R::Primitive<Float, D>;

    fn float_powi_scalar<const D: usize>(
        lhs: R::Primitive<Float, D>,
        rhs: FloatElem<B>,
    ) -> R::Primitive<Float, D>;

    fn float_clamp<const D: usize>(
        tensor: R::Primitive<Float, D>,
        min: FloatElem<B>,
        max: FloatElem<B>,
    ) -> R::Primitive<Float, D>;

    fn float_clamp_min<const D: usize>(
        tensor: R::Primitive<Float, D>,
        min: FloatElem<B>,
    ) -> R::Primitive<Float, D>;

    fn float_clamp_max<const D: usize>(
        tensor: R::Primitive<Float, D>,
        max: FloatElem<B>,
    ) -> R::Primitive<Float, D>;

    fn float_select<const D: usize>(
        tensor: R::Primitive<Float, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
    ) -> R::Primitive<Float, D>;

    fn float_select_assign<const D: usize>(
        tensor: R::Primitive<Float, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
        values: R::Primitive<Float, D>,
    ) -> R::Primitive<Float, D>;

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: R::Primitive<Float, D>,
        indices: IntTensor<B, D>,
    ) -> R::Primitive<Float, D>;

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: R::Primitive<Float, D>,
        indices: IntTensor<B, D>,
        values: R::Primitive<Float, D>,
    ) -> R::Primitive<Float, D>;

    fn float_sum<const D: usize>(tensor: R::Primitive<Float, D>) -> R::Primitive<Float, 1>;

    fn float_sum_dim<const D: usize>(
        tensor: R::Primitive<Float, D>,
        dim: usize,
    ) -> R::Primitive<Float, D>;

    fn float_prod_dim<const D: usize>(
        tensor: R::Primitive<Float, D>,
        dim: usize,
    ) -> R::Primitive<Float, D>;

    fn float_mean<const D: usize>(tensor: R::Primitive<Float, D>) -> R::Primitive<Float, 1>;

    fn float_mean_dim<const D: usize>(
        tensor: R::Primitive<Float, D>,
        dim: usize,
    ) -> R::Primitive<Float, D>;

    fn float_remainder_scalar<const D: usize>(
        lhs: R::Primitive<Float, D>,
        rhs: FloatElem<B>,
    ) -> R::Primitive<Float, D>;

    fn float_neg<const D: usize>(tensor: R::Primitive<Float, D>) -> R::Primitive<Float, D>;
}

pub trait SparseBoolOps<R: SparseRepr<B>, B: Backend> {
    fn bool_to_sparse<const D: usize>(dense: B::BoolTensorPrimitive<D>) -> R::Primitive<Bool, D>;

    fn bool_empty<const D: usize>(shape: Shape<D>, device: &Device<B>) -> R::Primitive<Bool, D>;

    fn bool_shape<const D: usize>(tensor: &R::Primitive<Bool, D>) -> Shape<D>;

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: R::Primitive<Bool, D1>,
        shape: Shape<D2>,
    ) -> R::Primitive<Bool, D2>;

    fn bool_transpose<const D: usize>(tensor: R::Primitive<Bool, D>) -> R::Primitive<Bool, D>;

    fn bool_swap_dims<const D: usize>(
        tensor: R::Primitive<Bool, D>,
        dim1: usize,
        dim2: usize,
    ) -> R::Primitive<Bool, D>;

    fn bool_permute<const D: usize>(
        tensor: R::Primitive<Bool, D>,
        axes: &[usize],
    ) -> R::Primitive<Bool, D>;

    fn bool_flip<const D: usize>(
        tensor: R::Primitive<Bool, D>,
        axes: &[usize],
    ) -> R::Primitive<Bool, D>;

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: R::Primitive<Bool, D1>,
        indices: [Range<usize>; D2],
    ) -> R::Primitive<Bool, D1>;

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: R::Primitive<Bool, D1>,
        ranges: [Range<usize>; D2],
        value: R::Primitive<Bool, D1>,
    ) -> R::Primitive<Bool, D1>;

    fn bool_device<const D: usize>(tensor: &R::Primitive<Bool, D>) -> Device<B>;

    fn bool_to_device<const D: usize>(
        tensor: R::Primitive<Bool, D>,
        device: &Device<B>,
    ) -> R::Primitive<Bool, D>;

    fn bool_into_data<const D: usize>(
        tensor: R::Primitive<Bool, D>,
    ) -> impl Future<Output = TensorData> + Send;

    fn bool_from_data<const D: usize>(
        data: TensorData,
        device: &Device<B>,
    ) -> R::Primitive<Bool, D>;

    fn bool_repeat_dim<const D: usize>(
        tensor: R::Primitive<Bool, D>,
        dim: usize,
        times: usize,
    ) -> R::Primitive<Bool, D>;

    fn bool_cat<const D: usize>(
        tensors: Vec<R::Primitive<Bool, D>>,
        dim: usize,
    ) -> R::Primitive<Bool, D>;

    fn bool_equal<const D: usize>(
        lhs: R::Primitive<Bool, D>,
        rhs: R::Primitive<Bool, D>,
    ) -> R::Primitive<Bool, D>;

    fn bool_not_equal<const D: usize>(
        lhs: R::Primitive<Bool, D>,
        rhs: R::Primitive<Bool, D>,
    ) -> R::Primitive<Bool, D>;

    fn bool_any<const D: usize>(tensor: R::Primitive<Bool, D>) -> R::Primitive<Bool, 1>;

    fn bool_any_dim<const D: usize>(
        tensor: R::Primitive<Bool, D>,
        dim: usize,
    ) -> R::Primitive<Bool, D>;

    fn bool_all<const D: usize>(tensor: R::Primitive<Bool, D>) -> R::Primitive<Bool, 1>;

    fn bool_all_dim<const D: usize>(
        tensor: R::Primitive<Bool, D>,
        dim: usize,
    ) -> R::Primitive<Bool, D>;

    fn bool_expand<const D1: usize, const D2: usize>(
        tensor: R::Primitive<Bool, D1>,
        shape: Shape<D2>,
    ) -> R::Primitive<Bool, D2>;
}

pub trait SparseIntOps<R: SparseRepr<B>, B: Backend> {
    fn int_empty<const D: usize>(shape: Shape<D>, device: &Device<B>) -> R::Primitive<Int, D>;

    fn int_shape<const D: usize>(tensor: &R::Primitive<Int, D>) -> Shape<D>;

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: R::Primitive<Int, D1>,
        shape: Shape<D2>,
    ) -> R::Primitive<Int, D2>;

    fn int_transpose<const D: usize>(tensor: R::Primitive<Int, D>) -> R::Primitive<Int, D>;

    fn int_swap_dims<const D: usize>(
        tensor: R::Primitive<Int, D>,
        dim1: usize,
        dim2: usize,
    ) -> R::Primitive<Int, D>;

    fn int_permute<const D: usize>(
        tensor: R::Primitive<Int, D>,
        axes: &[usize],
    ) -> R::Primitive<Int, D>;

    fn int_flip<const D: usize>(
        tensor: R::Primitive<Int, D>,
        axes: &[usize],
    ) -> R::Primitive<Int, D>;

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: R::Primitive<Int, D1>,
        indices: [Range<usize>; D2],
    ) -> R::Primitive<Int, D1>;

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: R::Primitive<Int, D1>,
        ranges: [Range<usize>; D2],
        value: R::Primitive<Int, D1>,
    ) -> R::Primitive<Int, D1>;

    fn int_device<const D: usize>(tensor: &R::Primitive<Int, D>) -> Device<B>;

    fn int_to_device<const D: usize>(
        tensor: R::Primitive<Int, D>,
        device: &Device<B>,
    ) -> R::Primitive<Int, D>;

    fn int_into_data<const D: usize>(
        tensor: R::Primitive<Int, D>,
    ) -> impl Future<Output = TensorData> + Send;

    fn int_from_data<const D: usize>(data: TensorData, device: &Device<B>) -> R::Primitive<Int, D>;

    fn int_repeat_dim<const D: usize>(
        tensor: R::Primitive<Int, D>,
        dim: usize,
        times: usize,
    ) -> R::Primitive<Int, D>;

    fn int_cat<const D: usize>(
        tensors: Vec<R::Primitive<Int, D>>,
        dim: usize,
    ) -> R::Primitive<Int, D>;

    fn int_equal<const D: usize>(
        lhs: R::Primitive<Int, D>,
        rhs: R::Primitive<Int, D>,
    ) -> R::Primitive<Bool, D>;

    fn int_not_equal<const D: usize>(
        lhs: R::Primitive<Int, D>,
        rhs: R::Primitive<Int, D>,
    ) -> R::Primitive<Bool, D>;

    fn int_any<const D: usize>(tensor: R::Primitive<Int, D>) -> R::Primitive<Bool, 1>;

    fn int_any_dim<const D: usize>(
        tensor: R::Primitive<Int, D>,
        dim: usize,
    ) -> R::Primitive<Bool, D>;

    fn int_all<const D: usize>(tensor: R::Primitive<Int, D>) -> R::Primitive<Bool, 1>;

    fn int_all_dim<const D: usize>(
        tensor: R::Primitive<Int, D>,
        dim: usize,
    ) -> R::Primitive<Bool, D>;

    fn int_expand<const D1: usize, const D2: usize>(
        tensor: R::Primitive<Int, D1>,
        shape: Shape<D2>,
    ) -> R::Primitive<Int, D2>;
}
