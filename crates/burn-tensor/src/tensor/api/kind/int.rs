// BasicOps: CreationOps + AssignOps + ComparisonOps + ReductionOps + ViewOps

use crate::{
    AssignOps, BasicOps, ComparisonOps, CreationOps, ElementConversion, Int, Numeric,
    NumericComparisonOps, NumericCreationOps, NumericReductionOps, ReductionOps, Shape, TensorData,
    ViewOps, backend::Backend,
};

// BasicOps: CreationOps + AssignOps + ComparisonOps + ReductionOps + ViewOps

impl<B: Backend> CreationOps<B> for Int {
    fn empty(shape: Shape, device: &B::Device) -> Self::Primitive {
        B::int_empty(shape, device)
    }

    fn zeros(shape: Shape, device: &B::Device) -> Self::Primitive {
        B::int_zeros(shape, device)
    }

    fn ones(shape: Shape, device: &B::Device) -> Self::Primitive {
        B::int_ones(shape, device)
    }

    fn full<E: ElementConversion>(
        shape: Shape,
        fill_value: E,
        device: &B::Device,
    ) -> Self::Primitive {
        B::int_full(shape, fill_value.elem(), device)
    }
}

impl<B: Backend> AssignOps<B> for Int {
    fn slice_assign(
        tensor: Self::Primitive,
        ranges: &[core::ops::Range<usize>],
        value: Self::Primitive,
    ) -> Self::Primitive {
        B::int_slice_assign(tensor, ranges, value)
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
        values: Self::Primitive,
    ) -> Self::Primitive {
        B::int_scatter(dim, tensor, indices, values)
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive {
        B::int_mask_where(tensor, mask, source)
    }

    fn mask_fill(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        value: Self::Elem,
    ) -> Self::Primitive {
        B::int_mask_fill(tensor, mask, value)
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: B::IntTensorPrimitive,
        values: Self::Primitive,
    ) -> Self::Primitive {
        B::int_select_assign(tensor, dim, indices, values)
    }
}

impl<B: Backend> ComparisonOps<B> for Int {
    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_equal(lhs, rhs)
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_not_equal(lhs, rhs)
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_equal_elem(lhs, rhs)
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_not_equal_elem(lhs, rhs)
    }
}

impl<B: Backend> ReductionOps<B> for Int {
    fn any(tensor: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        B::int_any(tensor)
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> <B as Backend>::BoolTensorPrimitive {
        B::int_any_dim(tensor, dim)
    }

    fn all(tensor: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        B::int_all(tensor)
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> <B as Backend>::BoolTensorPrimitive {
        B::int_all_dim(tensor, dim)
    }
}

impl<B: Backend> ViewOps<B> for Int {
    fn transpose(tensor: Self::Primitive) -> Self::Primitive {
        B::int_transpose(tensor)
    }

    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive {
        B::int_swap_dims(tensor, dim1, dim2)
    }

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        B::int_permute(tensor, axes)
    }

    fn slice(tensor: Self::Primitive, ranges: &[core::ops::Range<usize>]) -> Self::Primitive {
        B::int_slice(tensor, ranges)
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        B::int_expand(tensor, shape)
    }
}

impl<B: Backend> BasicOps<B> for Int {
    fn device(tensor: &Self::Primitive) -> <B as Backend>::Device {
        B::int_device(tensor)
    }

    fn to_device(tensor: Self::Primitive, device: &<B as Backend>::Device) -> Self::Primitive {
        B::int_to_device(tensor, device)
    }

    fn register_transaction(tr: &mut crate::Transaction<B>, tensor: Self::Primitive) {
        tr.register_int(tensor);
    }

    fn from_data(data: crate::TensorData, device: &<B as Backend>::Device) -> Self::Primitive {
        B::int_from_data(data.convert::<B::IntElem>(), device)
    }

    fn from_data_dtype(
        data: crate::TensorData,
        device: &<B as Backend>::Device,
        dtype: crate::DType,
    ) -> Self::Primitive {
        if !dtype.is_int() {
            panic!("Expected int dtype, got {dtype:?}")
        }

        B::int_from_data(data.convert_dtype(dtype), device)
    }

    async fn into_data_async(tensor: Self::Primitive) -> TensorData {
        B::int_into_data(tensor).await
    }

    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        B::int_reshape(tensor, shape)
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        B::int_flip(tensor, axes)
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        B::int_repeat_dim(tensor, dim, times)
    }

    fn cat(vectors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        B::int_cat(vectors, dim)
    }

    fn gather(
        dim: usize,
        tensor: Self::Primitive,
        indices: <B as Backend>::IntTensorPrimitive,
    ) -> Self::Primitive {
        B::int_gather(dim, tensor, indices)
    }

    fn select(
        tensor: Self::Primitive,
        dim: usize,
        indices: <B as Backend>::IntTensorPrimitive,
    ) -> Self::Primitive {
        B::int_select(tensor, dim, indices)
    }
}

// Numeric: BasicOps + NumericCreationOps + NumericComparisonOps + NumericReductionOps

impl<B: Backend> NumericCreationOps<B> for Int {
    fn random(
        shape: Shape,
        distribution: crate::Distribution,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive {
        B::int_random(shape, distribution, device)
    }
}

impl<B: Backend> NumericComparisonOps<B> for Int {
    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_greater(lhs, rhs)
    }

    fn greater_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_greater_elem(lhs, rhs)
    }

    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_greater_equal(lhs, rhs)
    }

    fn greater_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_greater_equal_elem(lhs, rhs)
    }

    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_lower(lhs, rhs)
    }

    fn lower_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_lower_elem(lhs, rhs)
    }

    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_lower_equal(lhs, rhs)
    }

    fn lower_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_lower_equal_elem(lhs, rhs)
    }
}

impl<B: Backend> NumericReductionOps<B> for Int {
    fn sum(tensor: Self::Primitive) -> Self::Primitive {
        B::int_sum(tensor)
    }

    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_sum_dim(tensor, dim)
    }

    fn prod(tensor: Self::Primitive) -> Self::Primitive {
        B::int_prod(tensor)
    }

    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_prod_dim(tensor, dim)
    }

    fn mean(tensor: Self::Primitive) -> Self::Primitive {
        B::int_mean(tensor)
    }

    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_mean_dim(tensor, dim)
    }

    fn argmax(tensor: Self::Primitive, dim: usize) -> B::IntTensorPrimitive {
        B::int_argmax(tensor, dim)
    }

    fn argmin(tensor: Self::Primitive, dim: usize) -> B::IntTensorPrimitive {
        B::int_argmin(tensor, dim)
    }

    fn max(tensor: Self::Primitive) -> Self::Primitive {
        B::int_max(tensor)
    }

    fn max_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_max_dim(tensor, dim)
    }

    fn max_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, B::IntTensorPrimitive) {
        B::int_max_dim_with_indices(tensor, dim)
    }

    fn max_abs(tensor: Self::Primitive) -> Self::Primitive {
        B::int_max_abs(tensor)
    }

    fn max_abs_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_max_abs_dim(tensor, dim)
    }

    fn min(tensor: Self::Primitive) -> Self::Primitive {
        B::int_min(tensor)
    }

    fn min_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_min_dim(tensor, dim)
    }

    fn min_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, B::IntTensorPrimitive) {
        B::int_min_dim_with_indices(tensor, dim)
    }
}

impl<B: Backend> Numeric<B> for Int {
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> B::IntTensorPrimitive {
        B::int_add(lhs, rhs)
    }

    fn add_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_add_scalar(lhs, rhs.elem())
    }

    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> B::IntTensorPrimitive {
        B::int_sub(lhs, rhs)
    }

    fn sub_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_sub_scalar(lhs, rhs.elem())
    }

    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> B::IntTensorPrimitive {
        B::int_div(lhs, rhs)
    }

    fn div_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_div_scalar(lhs, rhs.elem())
    }

    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_remainder(lhs, rhs)
    }

    fn remainder_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_remainder_scalar(lhs, rhs.elem())
    }

    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> B::IntTensorPrimitive {
        B::int_mul(lhs, rhs)
    }

    fn mul_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_mul_scalar(lhs, rhs.elem())
    }

    fn neg(tensor: Self::Primitive) -> Self::Primitive {
        B::int_neg(tensor)
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        B::int_sign(tensor)
    }

    fn clamp(tensor: Self::Primitive, min: B::IntElem, max: B::IntElem) -> Self::Primitive {
        B::int_clamp(tensor, min, max)
    }

    fn clamp_min(tensor: Self::Primitive, min: B::IntElem) -> Self::Primitive {
        B::int_clamp_min(tensor, min)
    }

    fn clamp_max(tensor: Self::Primitive, max: B::IntElem) -> Self::Primitive {
        B::int_clamp_max(tensor, max)
    }

    fn abs(tensor: Self::Primitive) -> Self::Primitive {
        B::int_abs(tensor)
    }

    fn powf(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_powf(lhs, B::int_into_float(rhs))
    }

    fn powf_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> B::IntTensorPrimitive {
        B::int_powf_scalar(lhs, rhs.elem())
    }

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_powi(lhs, rhs)
    }

    fn powi_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_powi_scalar(lhs, rhs.elem())
    }

    fn sort(tensor: Self::Primitive, dim: usize, descending: bool) -> Self::Primitive {
        B::int_sort(tensor, dim, descending)
    }

    fn sort_with_indices(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive, B::IntTensorPrimitive) {
        B::int_sort_with_indices(tensor, dim, descending)
    }

    fn argsort(tensor: Self::Primitive, dim: usize, descending: bool) -> B::IntTensorPrimitive {
        B::int_argsort(tensor, dim, descending)
    }
}
