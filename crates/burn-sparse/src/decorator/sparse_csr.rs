use crate::backend::SparseBackend;
use crate::backend::SparseTensor;
use crate::decorator::SparseCSR;
use crate::decorator::SparseDecorator;
use burn_tensor::backend::Backend;
use burn_tensor::ops::FloatElem;
use burn_tensor::ops::FloatTensor;
use core::marker::PhantomData;

#[derive(Debug, Default, Clone)]
pub struct SparseCSRTensor<B: Backend, const D: usize> {
    _b: PhantomData<B>,
}

impl<B> SparseBackend for SparseDecorator<B, SparseCSR>
where
    B: Backend,
{
    type SparseTensorPrimitive<const D: usize> = SparseCSRTensor<B, D>;

    fn sparse_empty<const D: usize>(
        _shape: burn_tensor::Shape<D>,
        _device: &burn_tensor::Device<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_to_sparse<const D: usize>(
        _dense: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }

    fn sparse_to_dense<const D: usize>(
        _sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D> {
        todo!()
    }

    fn sparse_spmm<const D: usize>(
        _lhs: Self::SparseTensorPrimitive<D>,
        _rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D> {
        todo!()
    }

    fn sparse_sddmm<const D: usize>(
        _lhs: Self::FloatTensorPrimitive<D>,
        _rhs: Self::FloatTensorPrimitive<D>,
        _sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }

    fn sparse_slice<const D1: usize, const D2: usize>(
        _tensor: SparseTensor<Self, D1>,
        _indices: [std::ops::Range<usize>; D2],
    ) -> SparseTensor<Self, D1> {
        todo!()
    }

    fn sparse_device<const D: usize>(_tensor: &SparseTensor<Self, D>) -> burn_tensor::Device<Self> {
        todo!()
    }

    fn sparse_to_device<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _device: &burn_tensor::Device<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_shape<const D: usize>(_tensor: &SparseTensor<Self, D>) -> burn_tensor::Shape<D> {
        todo!()
    }

    async fn sparse_into_data<const D: usize>(
        _tensor: SparseTensor<Self, D>,
    ) -> burn_tensor::TensorData { todo!() }

    fn sparse_from_data<const D: usize>(
        _data: burn_tensor::TensorData,
        _device: &burn_tensor::Device<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_reshape<const D1: usize, const D2: usize>(
        _tensor: SparseTensor<Self, D1>,
        _shape: burn_tensor::Shape<D2>,
    ) -> SparseTensor<Self, D2> {
        todo!()
    }

    fn sparse_transpose<const D: usize>(_tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_swap_dims<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim1: usize,
        _dim2: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_permute<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _axes: &[usize],
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_flip<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _axes: &[usize],
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_slice_assign<const D1: usize, const D2: usize>(
        _tensor: SparseTensor<Self, D1>,
        _ranges: [std::ops::Range<usize>; D2],
        _value: SparseTensor<Self, D1>,
    ) -> SparseTensor<Self, D1> {
        todo!()
    }

    fn sparse_repeat<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
        _times: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_cat<const D: usize>(
        _tensors: Vec<SparseTensor<Self, D>>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_equal<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_not_equal<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_any<const D: usize>(
        _tensor: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, 1> {
        todo!()
    }

    fn sparse_any_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_all<const D: usize>(
        _tensor: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, 1> {
        todo!()
    }

    fn sparse_all_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_expand<const D1: usize, const D2: usize>(
        _tensor: SparseTensor<Self, D1>,
        _shape: burn_tensor::Shape<D2>,
    ) -> SparseTensor<Self, D2> {
        todo!()
    }

    fn sparse_coalesce_sum<const D: usize>(
        _tensor: Self::SparseTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }

    fn sparse_nonzero<const D: usize>(_tensor: Self::SparseTensorPrimitive<D>) -> usize {
        todo!()
    }

    fn sparse_density<const D: usize>(_sparse: Self::SparseTensorPrimitive<D>) -> f32 {
        todo!()
    }

    fn sparse_add<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_add_scalar<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_add_dense<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn sparse_sub<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_sub_dense<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn sparse_sub_scalar<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_mul<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_mul_dense<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn sparse_mul_scalar<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_div<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_div_dense<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn sparse_div_scalar<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_max<const D: usize>(_tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        todo!()
    }

    fn sparse_max_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_min<const D: usize>(_tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        todo!()
    }

    fn sparse_min_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_greater<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_greater_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_greater_equal<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_greater_equal_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_lower<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_lower_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_lower_equal<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_lower_equal_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_abs<const D: usize>(_tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_powf<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_powi<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_powf_scalar<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_powi_scalar<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_clamp<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _min: FloatElem<Self>,
        _max: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_clamp_min<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _min: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_clamp_max<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _max: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_select<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
        _indices: burn_tensor::ops::IntTensor<Self, 1>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_select_assign<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
        _indices: burn_tensor::ops::IntTensor<Self, 1>,
        _values: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_gather<const D: usize>(
        _dim: usize,
        _tensor: SparseTensor<Self, D>,
        _indices: burn_tensor::ops::IntTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_scatter<const D: usize>(
        _dim: usize,
        _tensor: SparseTensor<Self, D>,
        _indices: burn_tensor::ops::IntTensor<Self, D>,
        _values: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_sum<const D: usize>(_tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        todo!()
    }

    fn sparse_sum_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_prod<const D: usize>(_tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        todo!()
    }

    fn sparse_prod_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_mean<const D: usize>(_tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        todo!()
    }

    fn sparse_mean_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_equal_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_not_equal_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_remainder_scalar<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_neg<const D: usize>(_tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_sign<const D: usize>(_tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_remove_zeros<const D: usize>(
        _tensor: Self::SparseTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }
}
