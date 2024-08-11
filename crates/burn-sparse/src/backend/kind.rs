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
        B::sparse_flip(tensor, axes)
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
        B::sparse_remainder_scalar(lhs, rhs.elem())
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
        B::sparse_neg(tensor)
    }

    fn sign<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::sparse_sign(tensor)
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
        B::sparse_sum(tensor)
    }

    fn sum_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::sparse_sum_dim(tensor, dim)
    }

    fn prod<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::sparse_prod(tensor)
    }

    fn prod_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::sparse_prod_dim(tensor, dim)
    }

    fn mean<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::sparse_mean(tensor)
    }

    fn mean_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::sparse_mean_dim(tensor, dim)
    }

    fn equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_equal_elem(lhs, rhs))
    }

    fn not_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_not_equal_elem(lhs, rhs))
    }

    fn greater<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_greater(lhs, rhs))
    }

    fn greater_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_greater_elem(lhs, rhs))
    }

    fn greater_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_greater_equal(lhs, rhs))
    }

    fn greater_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_greater_equal_elem(lhs, rhs))
    }

    fn lower<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_lower(lhs, rhs))
    }

    fn lower_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_lower_elem(lhs, rhs))
    }

    fn lower_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_lower_equal(lhs, rhs))
    }

    fn lower_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, burn_tensor::Bool> {
        Tensor::new(B::sparse_lower_equal_elem(lhs, rhs))
    }

    fn mask_where<const D: usize>(
        _tensor: Self::Primitive<D>,
        _mask: Tensor<B, D, burn_tensor::Bool>,
        _source: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        panic!("masking of sparse tensors is unsupported")
    }

    fn mask_fill<const D: usize>(
        _tensor: Self::Primitive<D>,
        _mask: Tensor<B, D, burn_tensor::Bool>,
        _value: Self::Elem,
    ) -> Self::Primitive<D> {
        panic!("masking of sparse tensors is unsupported")
    }

    fn gather<const D: usize>(
        dim: usize,
        tensor: Self::Primitive<D>,
        indices: Tensor<B, D, burn_tensor::Int>,
    ) -> Self::Primitive<D> {
        B::sparse_gather(dim, tensor, indices.into_primitive())
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: Self::Primitive<D>,
        indices: Tensor<B, D, burn_tensor::Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::sparse_scatter(dim, tensor, indices.into_primitive(), values)
    }

    fn select<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indices: Tensor<B, 1, burn_tensor::Int>,
    ) -> Self::Primitive<D> {
        B::sparse_select(tensor, dim, indices.into_primitive())
    }

    fn select_assign<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indices: Tensor<B, 1, burn_tensor::Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::sparse_select_assign(tensor, dim, indices.into_primitive(), values)
    }

    fn argmax<const D: usize>(
        _tensor: Self::Primitive<D>,
        _dim: usize,
    ) -> <B as Backend>::IntTensorPrimitive<D> {
        panic!("Argmax is unsupported for sparse tensors");
    }

    fn argmin<const D: usize>(
        _tensor: Self::Primitive<D>,
        _dim: usize,
    ) -> <B as Backend>::IntTensorPrimitive<D> {
        panic!("Argmin is unsupported for sparse tensors");
    }

    fn max<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::sparse_max(tensor)
    }

    fn max_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::sparse_max_dim(tensor, dim)
    }

    fn max_dim_with_indices<const D: usize>(
        _tensor: Self::Primitive<D>,
        _dim: usize,
    ) -> (Self::Primitive<D>, <B as Backend>::IntTensorPrimitive<D>) {
        todo!()
    }

    fn min<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::sparse_min(tensor)
    }

    fn min_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::sparse_min_dim(tensor, dim)
    }

    fn min_dim_with_indices<const D: usize>(
        _tensor: Self::Primitive<D>,
        _dim: usize,
    ) -> (Self::Primitive<D>, <B as Backend>::IntTensorPrimitive<D>) {
        todo!()
    }

    fn clamp<const D: usize>(
        tensor: Self::Primitive<D>,
        min: Self::Elem,
        max: Self::Elem,
    ) -> Self::Primitive<D> {
        B::sparse_clamp(tensor, min, max)
    }

    fn clamp_min<const D: usize>(
        tensor: Self::Primitive<D>,
        min: Self::Elem,
    ) -> Self::Primitive<D> {
        B::sparse_clamp_min(tensor, min)
    }

    fn clamp_max<const D: usize>(
        tensor: Self::Primitive<D>,
        max: Self::Elem,
    ) -> Self::Primitive<D> {
        B::sparse_clamp_max(tensor, max)
    }

    fn abs<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::sparse_abs(tensor)
    }

    fn powf<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::sparse_powf(lhs, rhs)
    }

    fn powi<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::sparse_powi(lhs, rhs)
    }

    fn powf_scalar<const D: usize, E: burn_tensor::ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::sparse_powf_scalar(lhs, rhs.elem())
    }

    fn powi_scalar<const D: usize, E: burn_tensor::ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::sparse_powi_scalar(lhs, rhs.elem())
    }

    fn random<const D: usize>(
        _shape: Shape<D>,
        _distribution: burn_tensor::Distribution,
        _device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        panic!("Random is unsupported for sparse tensors")
    }

    fn sort<const D: usize>(
        _tensor: Self::Primitive<D>,
        _dim: usize,
        _descending: bool,
    ) -> Self::Primitive<D> {
        panic!("Sorting is unsupported for sparse tensors")
    }

    fn sort_with_indices<const D: usize>(
        _tensor: Self::Primitive<D>,
        _dim: usize,
        _descending: bool,
    ) -> (
        Self::Primitive<D>,
        <burn_tensor::Int as TensorKind<B>>::Primitive<D>,
    ) {
        panic!("Sorting is unsupported for sparse tensors")
    }

    fn argsort<const D: usize>(
        _tensor: Self::Primitive<D>,
        _dim: usize,
        _descending: bool,
    ) -> <burn_tensor::Int as TensorKind<B>>::Primitive<D> {
        panic!("Sorting is unsupported for sparse tensors")
    }
}
