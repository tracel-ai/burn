use crate::{
    backend::Backend, check::TensorCheck, BasicOps, Bool, DType, Device, Element, Float, Int,
    Shape, Sparse, SparseRepr, Tensor, TensorData, TensorKind, TensorPrimitive, TensorRepr,
};
use core::{future::Future, ops::Range};

use crate::check;

impl<B: Backend, R: SparseRepr<B>> BasicOps<B, Sparse<R, B>> for Float {
    type Elem = B::FloatElem;

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<B as Backend>::Device,
    ) -> R::Primitive<Float, D> {
        R::float_empty(shape, device)
    }

    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        R::float_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        R::float_reshape(tensor, shape)
    }

    fn transpose<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        R::float_transpose(tensor)
    }

    fn swap_dims<const D: usize>(
        tensor: Self::Primitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> Self::Primitive<D> {
        R::float_swap_dims(tensor, dim1, dim2)
    }

    fn permute<const D: usize>(tensor: Self::Primitive<D>, axes: [usize; D]) -> Self::Primitive<D> {
        R::float_permute(tensor, &axes)
    }

    fn flip<const D: usize>(tensor: Self::Primitive<D>, axes: &[usize]) -> Self::Primitive<D> {
        R::float_flip(tensor, axes)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        range: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        R::float_slice(tensor, range)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        R::float_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        R::float_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        R::float_to_device(tensor, device)
    }

    fn into_data_async<const D: usize>(
        tensor: Self::Primitive<D>,
    ) -> impl Future<Output = TensorData> + Send {
        R::float_into_data(tensor)
    }

    fn from_data<const D: usize>(
        data: TensorData,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        R::float_from_data(data, device)
    }

    fn repeat_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D> {
        R::float_repeat_dim(tensor, dim, times)
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        R::float_cat(vectors, dim)
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        R::float_expand(tensor, shape)
    }

    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        Tensor::new(R::float_equal(lhs, rhs))
    }

    fn not_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        Tensor::new(R::float_not_equal(lhs, rhs))
    }

    fn any<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool, Sparse<R, B>> {
        Tensor::new(R::float_any(tensor))
    }

    fn any_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        Tensor::new(R::float_any_dim(tensor, dim))
    }

    fn all<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool, Sparse<R, B>> {
        Tensor::new(R::float_all(tensor))
    }

    fn all_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        Tensor::new(R::float_all_dim(tensor, dim))
    }
}

impl<B: Backend, R: SparseRepr<B>> BasicOps<B, Sparse<R, B>> for Bool {
    type Elem = bool;

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        R::bool_empty(shape, device)
    }

    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        R::bool_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        R::bool_reshape(tensor, shape)
    }

    fn transpose<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        R::bool_transpose(tensor)
    }

    fn swap_dims<const D: usize>(
        tensor: Self::Primitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> Self::Primitive<D> {
        R::bool_swap_dims(tensor, dim1, dim2)
    }

    fn permute<const D: usize>(tensor: Self::Primitive<D>, axes: [usize; D]) -> Self::Primitive<D> {
        R::bool_permute(tensor, &axes)
    }

    fn flip<const D: usize>(tensor: Self::Primitive<D>, axes: &[usize]) -> Self::Primitive<D> {
        R::bool_flip(tensor, axes)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        range: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        R::bool_slice(tensor, range)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        R::bool_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        R::bool_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        R::bool_to_device(tensor, device)
    }

    fn into_data_async<const D: usize>(
        tensor: Self::Primitive<D>,
    ) -> impl Future<Output = TensorData> + Send {
        R::bool_into_data(tensor)
    }

    fn from_data<const D: usize>(
        data: TensorData,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        R::bool_from_data(data, device)
    }

    fn repeat_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D> {
        R::bool_repeat_dim(tensor, dim, times)
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        R::bool_cat(vectors, dim)
    }

    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        panic!("Non-zero preserving operations are not supported for sparse tensors");
    }

    fn not_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        panic!("Non-zero preserving operations are not supported for sparse tensors");
    }

    fn any<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool, Sparse<R, B>> {
        Tensor::new(R::bool_any(tensor))
    }

    fn any_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        Tensor::new(R::bool_any_dim(tensor, dim))
    }

    fn all<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool, Sparse<R, B>> {
        Tensor::new(R::bool_all(tensor))
    }

    fn all_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        Tensor::new(R::bool_all_dim(tensor, dim))
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        R::bool_expand(tensor, shape)
    }
}

impl<B: Backend, R: SparseRepr<B>> BasicOps<B, Sparse<R, B>> for Int {
    type Elem = i32;

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        R::int_empty(shape, device)
    }

    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        R::int_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        R::int_reshape(tensor, shape)
    }

    fn transpose<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        R::int_transpose(tensor)
    }

    fn swap_dims<const D: usize>(
        tensor: Self::Primitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> Self::Primitive<D> {
        R::int_swap_dims(tensor, dim1, dim2)
    }

    fn permute<const D: usize>(tensor: Self::Primitive<D>, axes: [usize; D]) -> Self::Primitive<D> {
        R::int_permute(tensor, &axes)
    }

    fn flip<const D: usize>(tensor: Self::Primitive<D>, axes: &[usize]) -> Self::Primitive<D> {
        R::int_flip(tensor, axes)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        range: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        R::int_slice(tensor, range)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        R::int_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        R::int_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        R::int_to_device(tensor, device)
    }

    fn into_data_async<const D: usize>(
        tensor: Self::Primitive<D>,
    ) -> impl Future<Output = TensorData> + Send {
        R::int_into_data(tensor)
    }

    fn from_data<const D: usize>(
        data: TensorData,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        R::int_from_data(data, device)
    }

    fn repeat_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D> {
        R::int_repeat_dim(tensor, dim, times)
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        R::int_cat(vectors, dim)
    }

    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        panic!("Non-zero preserving operations are not supported for sparse tensors");
        Tensor::new(R::int_equal(lhs, rhs))
    }

    fn not_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        panic!("Non-zero preserving operations are not supported for sparse tensors");
        Tensor::new(R::int_not_equal(lhs, rhs))
    }

    fn any<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool, Sparse<R, B>> {
        Tensor::new(R::int_any(tensor))
    }

    fn any_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        Tensor::new(R::int_any_dim(tensor, dim))
    }

    fn all<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool, Sparse<R, B>> {
        Tensor::new(R::int_all(tensor))
    }

    fn all_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> Tensor<B, D, Bool, Sparse<R, B>> {
        Tensor::new(R::int_all_dim(tensor, dim))
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        R::int_expand(tensor, shape)
    }
}
