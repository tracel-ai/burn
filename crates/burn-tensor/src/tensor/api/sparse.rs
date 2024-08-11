use crate::{
    backend::Backend, check::TensorCheck, BasicOps, Bool, DType, Device, Float, Int, Shape, Sparse,
    SparseRepr, Tensor, TensorData, TensorKind, TensorPrimitive,
};
use core::{future::Future, ops::Range};

use crate::check;

type Primitive<K, B, R, const D: usize> = <K as TensorKind<B, Sparse<B, R>>>::Primitive<D>;

impl<B: Backend, R: SparseRepr<B>> BasicOps<B, Sparse<R, B>> for Float
where
    Float: TensorKind<B, Sparse<B, R>>,
{
    type Elem = B::FloatElem;

    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> Primitive<Self, B, R, D> {
        R::float_empty(shape, device)
    }

    fn shape<const D: usize>(tensor: &Primitive<Self, B, R, D>) -> Shape<D> {
        R::float_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Primitive<Self, B, R, D1>,
        shape: Shape<D2>,
    ) -> Primitive<Self, B, R, D2> {
        R::float_reshape(tensor, shape)
    }

    fn transpose<const D: usize>(tensor: Primitive<Self, B, R, D>) -> Primitive<Self, B, R, D> {
        R::float_transpose(tensor)
    }

    fn swap_dims<const D: usize>(
        tensor: Primitive<Self, B, R, D>,
        dim1: usize,
        dim2: usize,
    ) -> Primitive<Self, B, R, D> {
        check!(TensorCheck::swap_dims::<D>(dim1, dim2));
        R::float_swap_dims(tensor, dim1, dim2)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Primitive<Self, B, R, D1>,
        ranges: [Range<usize>; D2],
    ) -> Primitive<Self, B, R, D1> {
        R::float_slice(tensor, ranges)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Primitive<Self, B, R, D1>,
        ranges: [Range<usize>; D2],
        value: Primitive<Self, B, R, D1>,
    ) -> Primitive<Self, B, R, D1> {
        R::float_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(tensor: &Primitive<Self, B, R, D>) -> <B as Backend>::Device {
        R::float_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Primitive<Self, B, R, D>,
        device: &<B as Backend>::Device,
    ) -> Primitive<Self, B, R, D> {
        R::float_to_device(tensor, device)
    }

    fn from_data<const D: usize>(data: TensorData, device: &B::Device) -> Primitive<Self, B, R, D> {
        R::float_from_data(data, device)
    }

    fn repeat_dim<const D: usize>(
        tensor: Primitive<Self, B, R, D>,
        dim: usize,
        times: usize,
    ) -> Primitive<Self, B, R, D> {
        R::float_repeat_dim(tensor, dim, times)
    }

    fn cat<const D: usize>(
        vectors: Vec<Primitive<Self, B, R, D>>,
        dim: usize,
    ) -> Primitive<Self, B, R, D> {
        R::float_cat(vectors.into_iter().map(|tensor| tensor).collect(), dim)
    }

    fn equal<const D: usize>(
        lhs: Primitive<Self, B, R, D>,
        rhs: Primitive<Self, B, R, D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(R::float_equal(lhs, rhs))
    }

    fn not_equal<const D: usize>(
        lhs: Primitive<Self, B, R, D>,
        rhs: Primitive<Self, B, R, D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(R::float_not_equal(lhs, rhs))
    }

    fn any<const D: usize>(tensor: Primitive<Self, B, R, D>) -> Tensor<B, 1, Bool> {
        Tensor::new(R::float_any(tensor))
    }

    fn any_dim<const D: usize>(tensor: Primitive<Self, B, R, D>, dim: usize) -> Tensor<B, D, Bool> {
        Tensor::new(R::float_any_dim(tensor, dim))
    }

    fn all<const D: usize>(tensor: Primitive<Self, B, R, D>) -> Tensor<B, 1, Bool> {
        Tensor::new(R::float_all(tensor))
    }

    fn all_dim<const D: usize>(tensor: Primitive<Self, B, R, D>, dim: usize) -> Tensor<B, D, Bool> {
        Tensor::new(R::float_all_dim(tensor, dim))
    }

    fn permute<const D: usize>(
        tensor: Primitive<Self, B, R, D>,
        axes: [usize; D],
    ) -> Primitive<Self, B, R, D> {
        R::float_permute(tensor.tensor(), &axes)
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: Primitive<Self, B, R, D1>,
        shape: Shape<D2>,
    ) -> Primitive<Self, B, R, D2> {
        R::float_expand(tensor, shape)
    }

    fn flip<const D: usize>(
        tensor: Primitive<Self, B, R, D>,
        axes: &[usize],
    ) -> Primitive<Self, B, R, D> {
        R::float_flip(tensor, axes)
    }

    // fn into_data_async<const D: usize>(
    //     tensor: Self::Primitive<D>,
    // ) -> impl Future<Output = TensorData> + Send {
    //     todo!()
    // }

    fn into_data_async<const D: usize>(
        tensor: Primitive<Self, B, R, D>,
    ) -> impl Future<Output = TensorData> + Send {
        R::float_into_data(tensor)
    }
}
