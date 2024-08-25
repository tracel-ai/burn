use crate::{
    backend::Backend, check::TensorCheck, BasicOps, Bool, DType, Dense, Device, Element, Float,
    Int, ReprPrimitive, Shape, Sparse, SparseStorage, Tensor, TensorData, TensorKind,
    TensorPrimitive, TensorRepr, TensorStorage,
};
use core::{future::Future, ops::Range};

use crate::check;

pub trait BasicSparseOps<B: Backend, K: TensorKind<B>, SR: SparseStorage<B>>
where
    (B, K, Sparse<B, SR>): TensorRepr,
{
    fn into_dense<const D: usize>(
        tensor: ReprPrimitive<B, K, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, K, Dense, D>;

    fn into_sparse<const D: usize>(
        tensor: ReprPrimitive<B, K, Dense, D>,
    ) -> ReprPrimitive<B, K, Sparse<B, SR>, D>;
}

impl<B: Backend, SR: SparseStorage<B>> BasicSparseOps<B, Float, SR> for SR
where
    (B, Float, Sparse<B, SR>): TensorRepr,
{
    fn into_dense<const D: usize>(
        tensor: ReprPrimitive<B, Float, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Float, Dense, D> {
        TensorPrimitive::Float(SR::float_to_dense(tensor))
    }

    fn into_sparse<const D: usize>(
        tensor: ReprPrimitive<B, Float, Dense, D>,
    ) -> ReprPrimitive<B, Float, Sparse<B, SR>, D> {
        SR::float_to_sparse(tensor.tensor())
    }
}

impl<B: Backend, SR: SparseStorage<B>> BasicOps<B, Sparse<B, SR>> for Float {
    type Elem = B::FloatElem;

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<B as Backend>::Device,
    ) -> SR::SparsePrimitive<Float, D> {
        SR::float_empty(shape, device)
    }

    fn shape<const D: usize>(tensor: &ReprPrimitive<B, Self, Sparse<B, SR>, D>) -> Shape<D> {
        SR::float_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        shape: Shape<D2>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D2> {
        SR::float_reshape(tensor, shape)
    }

    fn transpose<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::float_transpose(tensor)
    }

    fn swap_dims<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim1: usize,
        dim2: usize,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::float_swap_dims(tensor, dim1, dim2)
    }

    fn permute<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        axes: [usize; D],
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::float_permute(tensor, &axes)
    }

    fn flip<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        axes: &[usize],
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::float_flip(tensor, axes)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        range: [Range<usize>; D2],
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D1> {
        SR::float_slice(tensor, range)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        ranges: [Range<usize>; D2],
        value: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D1> {
        SR::float_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(
        tensor: &ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> <B as Backend>::Device {
        SR::float_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        device: &<B as Backend>::Device,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::float_to_device(tensor, device)
    }

    fn into_data_async<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> impl Future<Output = TensorData> + Send {
        async {
            panic!("into_data not supported for sparse tensors, convert to dense first.");
        }
    }

    fn from_data<const D: usize>(
        data: TensorData,
        device: &<B as Backend>::Device,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        panic!("from_data not supported for sparse tensors, convert from dense..");
    }

    fn repeat_dim<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim: usize,
        times: usize,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::float_repeat_dim(tensor, dim, times)
    }

    fn cat<const D: usize>(
        vectors: Vec<ReprPrimitive<B, Self, Sparse<B, SR>, D>>,
        dim: usize,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::float_cat(vectors, dim)
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        shape: Shape<D2>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D2> {
        SR::float_expand(tensor, shape)
    }

    fn equal<const D: usize>(
        lhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        Tensor::new(SR::float_equal(lhs, rhs))
    }

    fn not_equal<const D: usize>(
        lhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        Tensor::new(SR::float_not_equal(lhs, rhs))
    }

    fn any<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, 1, Bool, Sparse<B, SR>> {
        Tensor::new(SR::float_any(tensor))
    }

    fn any_dim<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        Tensor::new(SR::float_any_dim(tensor, dim))
    }

    fn all<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, 1, Bool, Sparse<B, SR>> {
        Tensor::new(SR::float_all(tensor))
    }

    fn all_dim<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        Tensor::new(SR::float_all_dim(tensor, dim))
    }
}

impl<B: Backend, SR: SparseStorage<B>> BasicOps<B, Sparse<B, SR>> for Bool {
    type Elem = bool;

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<B as Backend>::Device,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::bool_empty(shape, device)
    }

    fn shape<const D: usize>(tensor: &ReprPrimitive<B, Self, Sparse<B, SR>, D>) -> Shape<D> {
        SR::bool_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        shape: Shape<D2>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D2> {
        SR::bool_reshape(tensor, shape)
    }

    fn transpose<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::bool_transpose(tensor)
    }

    fn swap_dims<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim1: usize,
        dim2: usize,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::bool_swap_dims(tensor, dim1, dim2)
    }

    fn permute<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        axes: [usize; D],
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::bool_permute(tensor, &axes)
    }

    fn flip<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        axes: &[usize],
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::bool_flip(tensor, axes)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        range: [Range<usize>; D2],
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D1> {
        SR::bool_slice(tensor, range)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        ranges: [Range<usize>; D2],
        value: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D1> {
        SR::bool_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(
        tensor: &ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> <B as Backend>::Device {
        SR::bool_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        device: &<B as Backend>::Device,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::bool_to_device(tensor, device)
    }

    fn into_data_async<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> impl Future<Output = TensorData> + Send {
        async {
            panic!("into_data not supported for sparse tensors, convert to dense first.");
        }
    }

    fn from_data<const D: usize>(
        data: TensorData,
        device: &<B as Backend>::Device,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        panic!("from_data not supported for sparse tensors, convert from dense..");
    }

    fn repeat_dim<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim: usize,
        times: usize,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::bool_repeat_dim(tensor, dim, times)
    }

    fn cat<const D: usize>(
        vectors: Vec<ReprPrimitive<B, Self, Sparse<B, SR>, D>>,
        dim: usize,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::bool_cat(vectors, dim)
    }

    fn equal<const D: usize>(
        lhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        panic!("Non-zero preserving operations are not supported for sparse tensors");
    }

    fn not_equal<const D: usize>(
        lhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        panic!("Non-zero preserving operations are not supported for sparse tensors");
    }

    fn any<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, 1, Bool, Sparse<B, SR>> {
        Tensor::new(SR::bool_any(tensor))
    }

    fn any_dim<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        Tensor::new(SR::bool_any_dim(tensor, dim))
    }

    fn all<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, 1, Bool, Sparse<B, SR>> {
        Tensor::new(SR::bool_all(tensor))
    }

    fn all_dim<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        Tensor::new(SR::bool_all_dim(tensor, dim))
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        shape: Shape<D2>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D2> {
        SR::bool_expand(tensor, shape)
    }
}

impl<B: Backend, SR: SparseStorage<B>> BasicOps<B, Sparse<B, SR>> for Int {
    type Elem = i32;

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<B as Backend>::Device,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::int_empty(shape, device)
    }

    fn shape<const D: usize>(tensor: &ReprPrimitive<B, Self, Sparse<B, SR>, D>) -> Shape<D> {
        SR::int_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        shape: Shape<D2>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D2> {
        SR::int_reshape(tensor, shape)
    }

    fn transpose<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::int_transpose(tensor)
    }

    fn swap_dims<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim1: usize,
        dim2: usize,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::int_swap_dims(tensor, dim1, dim2)
    }

    fn permute<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        axes: [usize; D],
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::int_permute(tensor, &axes)
    }

    fn flip<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        axes: &[usize],
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::int_flip(tensor, axes)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        range: [Range<usize>; D2],
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D1> {
        SR::int_slice(tensor, range)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        ranges: [Range<usize>; D2],
        value: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D1> {
        SR::int_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(
        tensor: &ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> <B as Backend>::Device {
        SR::int_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        device: &<B as Backend>::Device,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::int_to_device(tensor, device)
    }

    fn into_data_async<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> impl Future<Output = TensorData> + Send {
        async {
            panic!("into_data not supported for sparse tensors, convert to dense first.");
        }
    }

    fn from_data<const D: usize>(
        data: TensorData,
        device: &<B as Backend>::Device,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        panic!("from_data not supported for sparse tensors, convert from dense..");
    }

    fn repeat_dim<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim: usize,
        times: usize,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::int_repeat_dim(tensor, dim, times)
    }

    fn cat<const D: usize>(
        vectors: Vec<ReprPrimitive<B, Self, Sparse<B, SR>, D>>,
        dim: usize,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D> {
        SR::int_cat(vectors, dim)
    }

    fn equal<const D: usize>(
        lhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        panic!("Non-zero preserving operations are not supported for sparse tensors");
        Tensor::new(SR::int_equal(lhs, rhs))
    }

    fn not_equal<const D: usize>(
        lhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        rhs: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        panic!("Non-zero preserving operations are not supported for sparse tensors");
        Tensor::new(SR::int_not_equal(lhs, rhs))
    }

    fn any<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, 1, Bool, Sparse<B, SR>> {
        Tensor::new(SR::int_any(tensor))
    }

    fn any_dim<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        Tensor::new(SR::int_any_dim(tensor, dim))
    }

    fn all<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
    ) -> Tensor<B, 1, Bool, Sparse<B, SR>> {
        Tensor::new(SR::int_all(tensor))
    }

    fn all_dim<const D: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<B, SR>> {
        Tensor::new(SR::int_all_dim(tensor, dim))
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: ReprPrimitive<B, Self, Sparse<B, SR>, D1>,
        shape: Shape<D2>,
    ) -> ReprPrimitive<B, Self, Sparse<B, SR>, D2> {
        SR::int_expand(tensor, shape)
    }
}
