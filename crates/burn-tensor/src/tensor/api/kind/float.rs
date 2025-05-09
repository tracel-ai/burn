use core::ops::Range;

use crate::{
    AssignOps, BasicOps, ComparisonOps, CreationOps, DType, Device, ElementConversion, Float,
    Numeric, NumericComparisonOps, NumericCreationOps, NumericReductionOps, ReductionOps, Shape,
    TensorData, TensorPrimitive, Transaction, ViewOps, backend::Backend,
};

// BasicOps: CreationOps + AssignOps + ComparisonOps + ReductionOps + ViewOps

impl<B: Backend> CreationOps<B> for Float {
    fn empty(shape: Shape, device: &B::Device) -> Self::Primitive {
        TensorPrimitive::Float(B::float_empty(shape, device))
    }

    fn zeros(shape: Shape, device: &B::Device) -> Self::Primitive {
        TensorPrimitive::Float(B::float_zeros(shape, device))
    }

    fn ones(shape: Shape, device: &B::Device) -> Self::Primitive {
        TensorPrimitive::Float(B::float_ones(shape, device))
    }

    fn full<E: ElementConversion>(
        shape: Shape,
        fill_value: E,
        device: &B::Device,
    ) -> Self::Primitive {
        TensorPrimitive::Float(B::float_full(shape, fill_value.elem(), device))
    }
}

impl<B: Backend> AssignOps<B> for Float {
    fn slice_assign(
        tensor: Self::Primitive,
        ranges: &[Range<usize>],
        value: Self::Primitive,
    ) -> Self::Primitive {
        TensorPrimitive::Float(B::float_slice_assign(
            tensor.tensor(),
            ranges,
            value.tensor(),
        ))
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
        values: Self::Primitive,
    ) -> Self::Primitive {
        TensorPrimitive::Float(B::float_scatter(
            dim,
            tensor.tensor(),
            indices,
            values.tensor(),
        ))
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive {
        TensorPrimitive::Float(B::float_mask_where(tensor.tensor(), mask, source.tensor()))
    }

    fn mask_fill(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        value: Self::Elem,
    ) -> Self::Primitive {
        TensorPrimitive::Float(B::float_mask_fill(tensor.tensor(), mask, value))
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: B::IntTensorPrimitive,
        values: Self::Primitive,
    ) -> Self::Primitive {
        // Select assign is ambiguous for QFloat
        TensorPrimitive::Float(B::float_select_assign(
            tensor.tensor(),
            dim,
            indices,
            values.tensor(),
        ))
    }
}

impl<B: Backend> ComparisonOps<B> for Float {
    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_equal(lhs.tensor(), rhs.tensor())
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_not_equal(lhs.tensor(), rhs.tensor())
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::float_equal_elem(lhs.tensor(), rhs)
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::float_not_equal_elem(lhs.tensor(), rhs)
    }
}

impl<B: Backend> ReductionOps<B> for Float {
    fn any(tensor: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_any(tensor.tensor())
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> B::BoolTensorPrimitive {
        B::float_any_dim(tensor.tensor(), dim)
    }

    fn all(tensor: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_all(tensor.tensor())
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> B::BoolTensorPrimitive {
        B::float_all_dim(tensor.tensor(), dim)
    }
}

impl<B: Backend> ViewOps<B> for Float {
    fn transpose(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_transpose(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_transpose(tensor)),
        }
    }

    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_swap_dims(tensor, dim1, dim2))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_swap_dims(tensor, dim1, dim2))
            }
        }
    }

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_permute(tensor, axes))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_permute(tensor, axes)),
        }
    }

    fn slice(tensor: Self::Primitive, ranges: &[Range<usize>]) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_slice(tensor, ranges))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_slice(tensor, ranges)),
        }
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_expand(tensor, shape))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_expand(tensor, shape)),
        }
    }
}

impl<B: Backend> BasicOps<B> for Float {
    fn register_transaction(tr: &mut Transaction<B>, tensor: Self::Primitive) {
        tr.register_float(tensor);
    }

    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_reshape(tensor, shape))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_reshape(tensor, shape)),
        }
    }

    fn device(tensor: &Self::Primitive) -> Device<B> {
        match tensor {
            TensorPrimitive::Float(tensor) => B::float_device(tensor),
            TensorPrimitive::QFloat(tensor) => B::q_device(tensor),
        }
    }

    fn to_device(tensor: Self::Primitive, device: &Device<B>) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_to_device(tensor, device))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_to_device(tensor, device))
            }
        }
    }

    async fn into_data_async(tensor: Self::Primitive) -> TensorData {
        match tensor {
            TensorPrimitive::Float(tensor) => B::float_into_data(tensor).await,
            TensorPrimitive::QFloat(tensor) => B::q_into_data(tensor).await,
        }
    }

    fn from_data(data: TensorData, device: &B::Device) -> Self::Primitive {
        match data.dtype {
            DType::QFloat(_strategy) => TensorPrimitive::QFloat(B::q_from_data(data, device)),
            _ => TensorPrimitive::Float(B::float_from_data(data.convert::<B::FloatElem>(), device)),
        }
    }

    fn from_data_dtype(data: TensorData, device: &B::Device, dtype: DType) -> Self::Primitive {
        match dtype {
            DType::QFloat(_strategy) => {
                TensorPrimitive::QFloat(B::q_from_data(data.convert_dtype(dtype), device))
            }
            _ if dtype.is_float() => {
                TensorPrimitive::Float(B::float_from_data(data.convert_dtype(dtype), device))
            }
            _ => panic!("Expected float dtype, got {dtype:?}"),
        }
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_repeat_dim(tensor, dim, times))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_repeat_dim(tensor, dim, times))
            }
        }
    }

    fn cat(vectors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        match vectors.first().unwrap() {
            TensorPrimitive::Float(_) => TensorPrimitive::Float(B::float_cat(
                vectors.into_iter().map(|tensor| tensor.tensor()).collect(),
                dim,
            )),
            TensorPrimitive::QFloat(_) => TensorPrimitive::QFloat(B::q_cat(
                vectors
                    .into_iter()
                    .map(|tensor| {
                        if let TensorPrimitive::QFloat(t) = tensor {
                            t
                        } else {
                            panic!("Concatenation only works with vector of QFloat")
                        }
                    })
                    .collect(),
                dim,
            )),
        }
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_flip(tensor, axes)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_flip(tensor, axes)),
        }
    }

    fn gather(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
    ) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_gather(dim, tensor, indices))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_gather(dim, tensor, indices))
            }
        }
    }

    fn select(
        tensor: Self::Primitive,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_select(tensor, dim, indices))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_select(tensor, dim, indices))
            }
        }
    }
}

// Numeric: BasicOps + NumericCreationOps + NumericComparisonOps + NumericReductionOps

impl<B: Backend> NumericCreationOps<B> for Float {
    fn random(
        shape: Shape,
        distribution: crate::Distribution,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive {
        TensorPrimitive::Float(B::float_random(shape, distribution, device))
    }
}

impl<B: Backend> NumericComparisonOps<B> for Float {
    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        B::float_greater(lhs.tensor(), rhs.tensor())
    }

    fn greater_elem(lhs: Self::Primitive, rhs: Self::Elem) -> <B as Backend>::BoolTensorPrimitive {
        B::float_greater_elem(lhs.tensor(), rhs)
    }

    fn greater_equal(
        lhs: Self::Primitive,
        rhs: Self::Primitive,
    ) -> <B as Backend>::BoolTensorPrimitive {
        B::float_greater_equal(lhs.tensor(), rhs.tensor())
    }

    fn greater_equal_elem(
        lhs: Self::Primitive,
        rhs: Self::Elem,
    ) -> <B as Backend>::BoolTensorPrimitive {
        B::float_greater_equal_elem(lhs.tensor(), rhs)
    }

    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        B::float_lower(lhs.tensor(), rhs.tensor())
    }

    fn lower_elem(lhs: Self::Primitive, rhs: Self::Elem) -> <B as Backend>::BoolTensorPrimitive {
        B::float_lower_elem(lhs.tensor(), rhs)
    }

    fn lower_equal(
        lhs: Self::Primitive,
        rhs: Self::Primitive,
    ) -> <B as Backend>::BoolTensorPrimitive {
        B::float_lower_equal(lhs.tensor(), rhs.tensor())
    }

    fn lower_equal_elem(
        lhs: Self::Primitive,
        rhs: Self::Elem,
    ) -> <B as Backend>::BoolTensorPrimitive {
        B::float_lower_equal_elem(lhs.tensor(), rhs)
    }
}

impl<B: Backend> NumericReductionOps<B> for Float {
    fn sum(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_sum(tensor)),
            TensorPrimitive::QFloat(tensor) => B::q_sum(tensor),
        }
    }

    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_sum_dim(tensor, dim)),
            TensorPrimitive::QFloat(tensor) => B::q_sum_dim(tensor, dim),
        }
    }

    fn prod(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_prod(tensor)),
            TensorPrimitive::QFloat(tensor) => B::q_prod(tensor),
        }
    }

    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_prod_dim(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => B::q_prod_dim(tensor, dim),
        }
    }

    fn mean(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_mean(tensor)),
            TensorPrimitive::QFloat(tensor) => B::q_mean(tensor),
        }
    }

    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_mean_dim(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => B::q_mean_dim(tensor, dim),
        }
    }

    fn argmax(tensor: Self::Primitive, dim: usize) -> <B as Backend>::IntTensorPrimitive {
        match tensor {
            TensorPrimitive::Float(tensor) => B::float_argmax(tensor, dim),
            TensorPrimitive::QFloat(tensor) => B::q_argmax(tensor, dim),
        }
    }

    fn argmin(tensor: Self::Primitive, dim: usize) -> <B as Backend>::IntTensorPrimitive {
        match tensor {
            TensorPrimitive::Float(tensor) => B::float_argmin(tensor, dim),
            TensorPrimitive::QFloat(tensor) => B::q_argmin(tensor, dim),
        }
    }

    fn max(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_max(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_max(tensor)),
        }
    }

    fn max_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_max_dim(tensor, dim)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_max_dim(tensor, dim)),
        }
    }

    fn max_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, B::IntTensorPrimitive) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let (values, indices) = B::float_max_dim_with_indices(tensor, dim);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let (values, indices) = B::q_max_dim_with_indices(tensor, dim);
                (TensorPrimitive::QFloat(values), indices)
            }
        }
    }

    fn max_abs(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_max_abs(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_max_abs(tensor)),
        }
    }

    fn max_abs_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_max_abs_dim(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_max_abs_dim(tensor, dim))
            }
        }
    }

    fn min(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_min(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_min(tensor)),
        }
    }

    fn min_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_min_dim(tensor, dim)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_min_dim(tensor, dim)),
        }
    }

    fn min_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, B::IntTensorPrimitive) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let (values, indices) = B::float_min_dim_with_indices(tensor, dim);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let (values, indices) = B::q_min_dim_with_indices(tensor, dim);
                (TensorPrimitive::QFloat(values), indices)
            }
        }
    }
}

impl<B: Backend> Numeric<B> for Float {
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_add(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => B::q_add(lhs, rhs),
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }

    fn add_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_add_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => B::q_add_scalar(lhs, rhs.elem()),
        }
    }

    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_sub(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => B::q_sub(lhs, rhs),
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }

    fn sub_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_sub_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => B::q_sub_scalar(lhs, rhs.elem()),
        }
    }

    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_div(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => B::q_div(lhs, rhs),
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }

    fn div_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_div_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => B::q_div_scalar(lhs, rhs.elem()),
        }
    }

    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_remainder(lhs.tensor(), rhs.tensor()))
    }

    fn remainder_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        TensorPrimitive::Float(B::float_remainder_scalar(lhs.tensor(), rhs.elem()))
    }

    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_mul(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => B::q_mul(lhs, rhs),
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }

    fn mul_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_mul_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => B::q_mul_scalar(lhs, rhs.elem()),
        }
    }

    fn neg(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_neg(tensor)),
            TensorPrimitive::QFloat(tensor) => B::q_neg(tensor),
        }
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_sign(tensor.tensor()))
    }

    fn clamp(tensor: Self::Primitive, min: B::FloatElem, max: B::FloatElem) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_clamp(tensor, min, max))
            }
            TensorPrimitive::QFloat(tensor) => B::q_clamp(tensor, min, max),
        }
    }

    fn clamp_min(tensor: Self::Primitive, min: B::FloatElem) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_clamp_min(tensor, min))
            }
            TensorPrimitive::QFloat(tensor) => B::q_clamp_min(tensor, min),
        }
    }

    fn clamp_max(tensor: Self::Primitive, max: B::FloatElem) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_clamp_max(tensor, max))
            }
            TensorPrimitive::QFloat(tensor) => B::q_clamp_max(tensor, max),
        }
    }

    fn abs(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_abs(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_abs(tensor)),
        }
    }

    fn powf(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_powf(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => B::q_powf(lhs, rhs),
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }

    fn powf_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_powf_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => B::q_powf_scalar(lhs, rhs.elem()),
        }
    }

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_powf(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => B::q_powf(lhs, rhs),
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }

    fn powi_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_powi_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => B::q_powi_scalar(lhs, rhs.elem()),
        }
    }

    fn sort(tensor: Self::Primitive, dim: usize, descending: bool) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_sort(tensor, dim, descending))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_sort(tensor, dim, descending))
            }
        }
    }

    fn sort_with_indices(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive, B::IntTensorPrimitive) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let (values, indices) = B::float_sort_with_indices(tensor, dim, descending);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let (values, indices) = B::q_sort_with_indices(tensor, dim, descending);
                (TensorPrimitive::QFloat(values), indices)
            }
        }
    }

    fn argsort(tensor: Self::Primitive, dim: usize, descending: bool) -> B::IntTensorPrimitive {
        match tensor {
            TensorPrimitive::Float(tensor) => B::float_argsort(tensor, dim, descending),
            TensorPrimitive::QFloat(tensor) => B::q_argsort(tensor, dim, descending),
        }
    }
}
