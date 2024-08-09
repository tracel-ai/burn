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
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_to_sparse<const D: usize>(
        dense: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }

    fn sparse_to_dense<const D: usize>(
        sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D> {
        todo!()
    }

    fn sparse_spmm<const D: usize>(
        lhs: Self::SparseTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D> {
        todo!()
    }

    fn sparse_sddmm<const D: usize>(
        lhs: Self::FloatTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
        sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }

    fn sparse_slice<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> SparseTensor<Self, D1> {
        todo!()
    }

    fn sparse_device<const D: usize>(tensor: &SparseTensor<Self, D>) -> burn_tensor::Device<Self> {
        todo!()
    }

    fn sparse_to_device<const D: usize>(
        tensor: SparseTensor<Self, D>,
        device: &burn_tensor::Device<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_shape<const D: usize>(tensor: &SparseTensor<Self, D>) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn sparse_into_data<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> impl std::future::Future<Output = burn_tensor::TensorData> + Send {
        async { todo!() }
    }

    fn sparse_from_data<const D: usize>(
        data: burn_tensor::TensorData,
        device: &burn_tensor::Device<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_reshape<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> SparseTensor<Self, D2> {
        todo!()
    }

    fn sparse_transpose<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_swap_dims<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_permute<const D: usize>(
        tensor: SparseTensor<Self, D>,
        axes: &[usize],
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_flip<const D: usize>(
        tensor: SparseTensor<Self, D>,
        axes: &[usize],
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_slice_assign<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: SparseTensor<Self, D1>,
    ) -> SparseTensor<Self, D1> {
        todo!()
    }

    fn sparse_repeat<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_cat<const D: usize>(
        tensors: Vec<SparseTensor<Self, D>>,
        dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_not_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_any<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, 1> {
        todo!()
    }

    fn sparse_any_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_all<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, 1> {
        todo!()
    }

    fn sparse_all_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_expand<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> SparseTensor<Self, D2> {
        todo!()
    }

    fn sparse_coalesce_sum<const D: usize>(
        tensor: Self::SparseTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }

    fn sparse_nonzero<const D: usize>(tensor: Self::SparseTensorPrimitive<D>) -> usize {
        todo!()
    }

    fn sparse_density<const D: usize>(sparse: Self::SparseTensorPrimitive<D>) -> f32 {
        todo!()
    }

    fn sparse_add<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_add_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_add_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn sparse_sub<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_sub_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn sparse_sub_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_mul<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_mul_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn sparse_mul_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_div<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_div_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn sparse_div_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_max<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        todo!()
    }

    fn sparse_max_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_min<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        todo!()
    }

    fn sparse_min_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_greater<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_greater_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_greater_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_greater_equal_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_lower<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_lower_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_lower_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_lower_equal_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_abs<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_powf<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_powi<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_powf_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_powi_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_clamp<const D: usize>(
        tensor: SparseTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_clamp_min<const D: usize>(
        tensor: SparseTensor<Self, D>,
        min: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_clamp_max<const D: usize>(
        tensor: SparseTensor<Self, D>,
        max: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_select<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
        indices: burn_tensor::ops::IntTensor<Self, 1>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_select_assign<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
        indices: burn_tensor::ops::IntTensor<Self, 1>,
        values: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_gather<const D: usize>(
        dim: usize,
        tensor: SparseTensor<Self, D>,
        indices: burn_tensor::ops::IntTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_scatter<const D: usize>(
        dim: usize,
        tensor: SparseTensor<Self, D>,
        indices: burn_tensor::ops::IntTensor<Self, D>,
        values: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_sum<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        todo!()
    }

    fn sparse_sum_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_prod<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        todo!()
    }

    fn sparse_prod_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_mean<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        todo!()
    }

    fn sparse_mean_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_equal_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_not_equal_elem<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_remainder_scalar<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_neg<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_sign<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_remove_zeros<const D: usize>(
        tensor: Self::SparseTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }
}
