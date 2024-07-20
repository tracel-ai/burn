use std::{future::Future, ops::Range};

use crate::backend::SparseBackend;
use burn_tensor::{backend::Backend, BasicOps, Numeric, Shape, Tensor, TensorData, TensorKind};

/// A type-level representation of the kind of a sparse (float) tensor.
#[derive(Clone, Debug)]
pub struct Sparse;

impl<B: SparseBackend> TensorKind<B> for Sparse {
    type Primitive<const D: usize> = B::SparseTensorPrimitive<D>;
    fn name() -> &'static str {
        "Sparse"
    }
}

impl<B: SparseBackend> BasicOps<B> for Sparse {
    type Elem = B::FloatElem;

    fn into_data_async<const D: usize>(
        tensor: Self::Primitive<D>,
    ) -> impl Future<Output = TensorData> + Send {
        B::sparse_into_data(tensor)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        B::sparse_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_to_device(tensor, device)
    }

    fn from_data<const D: usize>(
        data: TensorData,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_from_data(data, device)
    }

    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        B::sparse_shape(tensor)
    }

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_empty(shape, device)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::sparse_slice(tensor, ranges)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::sparse_reshape(tensor, shape)
    }

    fn transpose<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::sparse_transpose(tensor)
    }

    fn swap_dims<const D: usize>(
        tensor: Self::Primitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> Self::Primitive<D> {
        B::sparse_swap_dims(tensor, dim1, dim2)
    }

    fn permute<const D: usize>(tensor: Self::Primitive<D>, axes: [usize; D]) -> Self::Primitive<D> {
        B::sparse_permute(tensor, &axes)
    }

    fn flip<const D: usize>(tensor: Self::Primitive<D>, axes: &[usize]) -> Self::Primitive<D> {
        B::sparse_flip(tensor, &axes)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        B::sparse_slice_assign(tensor, ranges, value)
    }

    fn repeat<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D> {
        B::sparse_repeat(tensor, dim, times)
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        B::sparse_cat(vectors, dim)
    }

    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> burn_tensor::Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_equal(lhs, rhs))
    }

    fn not_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> burn_tensor::Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_not_equal(lhs, rhs))
    }

    fn any<const D: usize>(
        tensor: Self::Primitive<D>,
    ) -> burn_tensor::Tensor<B, 1, burn_tensor::Bool> {
        Tensor::new(B::sparse_any(tensor))
    }

    fn any_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> burn_tensor::Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_any_dim(tensor, dim))
    }

    fn all<const D: usize>(
        tensor: Self::Primitive<D>,
    ) -> burn_tensor::Tensor<B, 1, burn_tensor::Bool> {
        Tensor::new(B::sparse_all(tensor))
    }

    fn all_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> burn_tensor::Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_all_dim(tensor, dim))
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::sparse_expand(tensor, shape)
    }
}

impl<B: SparseBackend> Numeric<B> for Sparse {
    fn add<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D> {
        B::sparse_add(lhs, rhs)
    }

    fn add_scalar<const D: usize, E: burn_tensor::ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::sparse_add_scalar(lhs, rhs.elem())
    }

    fn sub<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D> {
        B::sparse_sub(lhs, rhs)
    }

    fn sub_scalar<const D: usize, E: burn_tensor::ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::sparse_sub_scalar(lhs, rhs.elem())
    }

    fn div<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D> {
        B::sparse_div(lhs, rhs)
    }

    fn div_scalar<const D: usize, E: burn_tensor::ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::sparse_div_scalar(lhs, rhs.elem())
    }

    fn remainder_scalar<const D: usize, E: burn_tensor::ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn mul<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D> {
        B::sparse_mul(lhs, rhs)
    }

    fn mul_scalar<const D: usize, E: burn_tensor::ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::sparse_mul_scalar(lhs, rhs.elem())
    }

    fn neg<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        todo!()
    }

    fn sign<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        todo!()
    }

    fn zeros<const D: usize>(
        shape: Shape<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_empty(shape, device)
    }

    fn ones<const D: usize>(
        shape: Shape<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_to_sparse(B::float_ones(shape, device))
    }

    fn full<const D: usize, E: burn_tensor::ElementConversion>(
        shape: Shape<D>,
        fill_value: E,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_to_sparse(B::float_full(shape, fill_value.elem(), device))
    }

    fn sum<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        todo!()
    }

    fn sum_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        todo!()
    }

    fn prod<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        todo!()
    }

    fn prod_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        todo!()
    }

    fn mean<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        todo!()
    }

    fn mean_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        todo!()
    }

    fn equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn not_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn greater<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn greater_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn greater_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn greater_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn lower<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn lower_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn lower_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn lower_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn mask_where<const D: usize>(
        tensor: Self::Primitive<D>,
        mask: Tensor<B, D, burn_tensor::Bool>,
        source: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn mask_fill<const D: usize>(
        tensor: Self::Primitive<D>,
        mask: Tensor<B, D, burn_tensor::Bool>,
        value: Self::Elem,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn gather<const D: usize>(
        dim: usize,
        tensor: Self::Primitive<D>,
        indices: Tensor<B, D, burn_tensor::Int>,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: Self::Primitive<D>,
        indices: Tensor<B, D, burn_tensor::Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn select<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indices: Tensor<B, 1, burn_tensor::Int>,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn select_assign<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indices: Tensor<B, 1, burn_tensor::Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn argmax<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> <B as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn argmin<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> <B as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn max<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        todo!()
    }

    fn max_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        todo!()
    }

    fn max_dim_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> (Self::Primitive<D>, <B as Backend>::IntTensorPrimitive<D>) {
        todo!()
    }

    fn min<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        todo!()
    }

    fn min_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        todo!()
    }

    fn min_dim_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> (Self::Primitive<D>, <B as Backend>::IntTensorPrimitive<D>) {
        todo!()
    }

    fn clamp<const D: usize>(
        tensor: Self::Primitive<D>,
        min: Self::Elem,
        max: Self::Elem,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn clamp_min<const D: usize>(
        tensor: Self::Primitive<D>,
        min: Self::Elem,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn clamp_max<const D: usize>(
        tensor: Self::Primitive<D>,
        max: Self::Elem,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn abs<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        todo!()
    }

    fn powf<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn powi<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn powf_scalar<const D: usize, E: burn_tensor::ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn powi_scalar<const D: usize, E: burn_tensor::ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: burn_tensor::Distribution,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn sort<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn sort_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> (
        Self::Primitive<D>,
        <burn_tensor::Int as TensorKind<B>>::Primitive<D>,
    ) {
        todo!()
    }

    fn argsort<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> <burn_tensor::Int as TensorKind<B>>::Primitive<D> {
        todo!()
    }
}
