use burn_tensor::{backend::Backend, ops::SparseFloatOps, SparseRepr};

use super::coo::COO;
type R = COO;

impl<B: Backend> SparseFloatOps<R, B> for R {
    fn float_to_sparse<const D: usize>(
        dense: <B as burn_tensor::backend::Backend>::FloatTensorPrimitive<D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_to_dense<const D: usize>(
        sparse: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_spmm<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <B as burn_tensor::backend::Backend>::FloatTensorPrimitive<D>,
    ) -> <B as burn_tensor::backend::Backend>::FloatTensorPrimitive<D> {
        todo!()
    }

    fn float_sddmm<const D: usize>(
        lhs: <B as burn_tensor::backend::Backend>::FloatTensorPrimitive<D>,
        rhs: <B as burn_tensor::backend::Backend>::FloatTensorPrimitive<D>,
        sparse: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_coalesce_sum<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_remove_zeros<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_nonzero<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> usize {
        todo!()
    }

    fn float_density<const D: usize>(
        sparse: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> f32 {
        todo!()
    }

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D1> {
        todo!()
    }

    fn float_device<const D: usize>(
        tensor: &<R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> burn_tensor::Device<B> {
        todo!()
    }

    fn float_to_device<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        device: &burn_tensor::Device<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_shape<const D: usize>(
        tensor: &<R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn float_into_data<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> impl std::future::Future<Output = burn_tensor::TensorData> + Send {
        async { todo!() }
    }

    fn float_from_data<const D: usize>(
        data: burn_tensor::TensorData,
        device: &burn_tensor::Device<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D2> {
        todo!()
    }

    fn float_transpose<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_swap_dims<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        dim1: usize,
        dim2: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_permute<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        axes: &[usize],
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_flip<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        axes: &[usize],
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D1>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D1> {
        todo!()
    }

    fn float_repeat_dim<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        dim: usize,
        times: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_cat<const D: usize>(
        tensors: Vec<<R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>>,
        dim: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_equal<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_not_equal<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_any<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, 1> {
        todo!()
    }

    fn float_any_dim<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_all<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, 1> {
        todo!()
    }

    fn float_all_dim<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D2> {
        todo!()
    }

    fn float_add<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_add_dense<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        todo!()
    }

    fn float_add_scalar<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_sub<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_sub_dense<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        todo!()
    }

    fn float_sub_scalar<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_mul<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_mul_dense<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        todo!()
    }

    fn float_mul_scalar<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_div<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_div_dense<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        todo!()
    }

    fn float_div_scalar<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_max<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_max_dim<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_min<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_min_dim<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_greater<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_greater_elem<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_greater_equal<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_greater_equal_elem<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_lower<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_lower_elem<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_lower_equal<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_lower_equal_elem<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_abs<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_sign<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_powf<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_powi<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_powf_scalar<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_powi_scalar<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_clamp<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        min: burn_tensor::ops::FloatElem<B>,
        max: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_clamp_min<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        min: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_clamp_max<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        max: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_select<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        dim: usize,
        indices: burn_tensor::ops::IntTensor<B, 1>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_select_assign<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        dim: usize,
        indices: burn_tensor::ops::IntTensor<B, 1>,
        values: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        indices: burn_tensor::ops::IntTensor<B, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        indices: burn_tensor::ops::IntTensor<B, D>,
        values: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_sum<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_sum_dim<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_prod<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_prod_dim<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_mean<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_mean_dim<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_equal_elem<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_not_equal_elem<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_remainder_scalar<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_neg<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Float, D> {
        todo!()
    }
}
