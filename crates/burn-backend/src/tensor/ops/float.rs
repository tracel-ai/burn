use alloc::vec::Vec;
use burn_std::{DType, Shape, Slice};

use crate::{
    AutodiffBackend, Backend, Distribution, ExecutionError, Scalar, TensorData, TensorPrimitive,
    ops::TransactionPrimitive,
    tensor::{
        BasicAutodiffOps, BasicOps, Device, Float, IndexingUpdateOp, IntTensor, Numeric, Ordered,
        TensorKind,
    },
};

macro_rules! q_bin_ops {
    ($lhs:ident, $rhs:ident, $op:ident, $q_op:ident) => {
        match ($lhs, $rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::$op(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => B::$q_op(lhs, rhs),
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::$op(B::dequantize(lhs), rhs))
            }
            (TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs)) => {
                TensorPrimitive::Float(B::$op(lhs, B::dequantize(rhs)))
            }
        }
    };
}

impl<B: Backend> BasicOps<B> for Float {
    type Elem = B::FloatElem;

    fn empty(shape: Shape, device: &Device<B>, dtype: DType) -> Self::Primitive {
        TensorPrimitive::Float(B::float_empty(shape, device, dtype.into()))
    }

    fn zeros(shape: Shape, device: &Device<B>, dtype: DType) -> Self::Primitive {
        TensorPrimitive::Float(B::float_zeros(shape, device, dtype.into()))
    }
    fn ones(shape: Shape, device: &Device<B>, dtype: DType) -> Self::Primitive {
        TensorPrimitive::Float(B::float_ones(shape, device, dtype.into()))
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device<B>, dtype: DType) -> Self::Primitive {
        TensorPrimitive::Float(B::float_full(shape, fill_value, device, dtype.into()))
    }

    fn register_transaction(tr: &mut TransactionPrimitive<B>, tensor: Self::Primitive) {
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

    fn slice(tensor: Self::Primitive, slices: &[Slice]) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_slice(tensor, slices))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_slice(tensor, slices)),
        }
    }

    fn slice_assign(
        tensor: Self::Primitive,
        slices: &[Slice],
        value: Self::Primitive,
    ) -> Self::Primitive {
        TensorPrimitive::Float(B::float_slice_assign(
            tensor.tensor(),
            slices,
            value.tensor(),
        ))
    }

    fn select(tensor: Self::Primitive, dim: usize, indices: IntTensor<B>) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_select(tensor, dim, indices))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_select(tensor, dim, indices))
            }
        }
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: IntTensor<B>,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        // Select assign is ambiguous for QFloat
        match update {
            IndexingUpdateOp::Add => TensorPrimitive::Float(B::float_select_add(
                tensor.tensor(),
                dim,
                indices,
                values.tensor(),
            )),
        }
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
        value: Scalar,
    ) -> Self::Primitive {
        TensorPrimitive::Float(B::float_mask_fill(tensor.tensor(), mask, value))
    }

    fn gather(dim: usize, tensor: Self::Primitive, indices: IntTensor<B>) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_gather(dim, tensor, indices))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_gather(dim, tensor, indices))
            }
        }
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: IntTensor<B>,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => TensorPrimitive::Float(B::float_scatter_add(
                dim,
                tensor.tensor(),
                indices,
                values.tensor(),
            )),
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

    async fn into_data_async(tensor: Self::Primitive) -> Result<TensorData, ExecutionError> {
        match tensor {
            TensorPrimitive::Float(tensor) => B::float_into_data(tensor).await,
            TensorPrimitive::QFloat(tensor) => B::q_into_data(tensor).await,
        }
    }

    fn from_data(data: TensorData, device: &Device<B>) -> Self::Primitive {
        match &data.dtype {
            DType::QFloat(_scheme) => TensorPrimitive::QFloat(B::q_from_data(data, device)),
            _ => TensorPrimitive::Float(B::float_from_data(data.convert::<B::FloatElem>(), device)),
        }
    }

    fn from_data_dtype(data: TensorData, device: &Device<B>, dtype: DType) -> Self::Primitive {
        match dtype {
            DType::QFloat(_scheme) => {
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

    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_equal(lhs.tensor(), rhs.tensor())
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_not_equal(lhs.tensor(), rhs.tensor())
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::float_equal_elem(lhs.tensor(), rhs)
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::float_not_equal_elem(lhs.tensor(), rhs)
    }

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

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_permute(tensor, axes))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_permute(tensor, axes)),
        }
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        TensorPrimitive::Float(B::float_expand(tensor.tensor(), shape))
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_flip(tensor, axes)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_flip(tensor, axes)),
        }
    }

    fn unfold(tensor: Self::Primitive, dim: usize, size: usize, step: usize) -> Self::Primitive {
        TensorPrimitive::Float(B::float_unfold(tensor.tensor(), dim, size, step))
    }
}

impl<B: Backend> Numeric<B> for Float {
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_add, q_add)
    }

    fn add_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => TensorPrimitive::Float(B::float_add_scalar(lhs, rhs)),
            TensorPrimitive::QFloat(lhs) => B::q_add_scalar(lhs, rhs),
        }
    }

    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_sub, q_sub)
    }

    fn sub_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => TensorPrimitive::Float(B::float_sub_scalar(lhs, rhs)),
            TensorPrimitive::QFloat(lhs) => B::q_sub_scalar(lhs, rhs),
        }
    }

    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_div, q_div)
    }

    fn div_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => TensorPrimitive::Float(B::float_div_scalar(lhs, rhs)),
            TensorPrimitive::QFloat(lhs) => B::q_div_scalar(lhs, rhs),
        }
    }
    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_remainder(lhs.tensor(), rhs.tensor()))
    }

    fn remainder_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        TensorPrimitive::Float(B::float_remainder_scalar(lhs.tensor(), rhs))
    }

    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_mul, q_mul)
    }

    fn mul_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => TensorPrimitive::Float(B::float_mul_scalar(lhs, rhs)),
            TensorPrimitive::QFloat(lhs) => B::q_mul_scalar(lhs, rhs),
        }
    }
    fn neg(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_neg(tensor)),
            TensorPrimitive::QFloat(tensor) => B::q_neg(tensor),
        }
    }

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

    fn cumsum(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_cumsum(tensor, dim)),
            TensorPrimitive::QFloat(tensor) => B::q_cumsum(tensor, dim),
        }
    }

    fn cumprod(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_cumprod(tensor, dim)),
            TensorPrimitive::QFloat(tensor) => B::q_cumprod(tensor, dim),
        }
    }

    fn abs(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_abs(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_abs(tensor)),
        }
    }

    fn powf(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_powf, q_powf)
    }

    fn powf_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => TensorPrimitive::Float(B::float_powf_scalar(lhs, rhs)),
            TensorPrimitive::QFloat(lhs) => B::q_powf_scalar(lhs, rhs),
        }
    }

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_powf, q_powf)
    }

    fn powi_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => TensorPrimitive::Float(B::float_powi_scalar(lhs, rhs)),
            TensorPrimitive::QFloat(lhs) => B::q_powi_scalar(lhs, rhs),
        }
    }

    fn random(shape: Shape, distribution: Distribution, device: &Device<B>) -> Self::Primitive {
        TensorPrimitive::Float(B::float_random(shape, distribution, device))
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_sign(tensor.tensor()))
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// # Panics
    ///
    /// If the two tensors don't have a compatible shape.
    fn matmul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_matmul(lhs, rhs))
            }
            (lhs, rhs) => B::q_matmul(lhs, rhs),
        }
    }
}
impl<B: Backend> Ordered<B> for Float {
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
    ) -> (Self::Primitive, IntTensor<B>) {
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

    fn argsort(tensor: Self::Primitive, dim: usize, descending: bool) -> IntTensor<B> {
        match tensor {
            TensorPrimitive::Float(tensor) => B::float_argsort(tensor, dim, descending),
            TensorPrimitive::QFloat(tensor) => B::q_argsort(tensor, dim, descending),
        }
    }

    fn cummin(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_cummin(tensor, dim)),
            TensorPrimitive::QFloat(tensor) => B::q_cummin(tensor, dim),
        }
    }

    fn cummax(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_cummax(tensor, dim)),
            TensorPrimitive::QFloat(tensor) => B::q_cummax(tensor, dim),
        }
    }

    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_greater(lhs.tensor(), rhs.tensor())
    }

    fn greater_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::float_greater_elem(lhs.tensor(), rhs)
    }

    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_greater_equal(lhs.tensor(), rhs.tensor())
    }

    fn greater_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::float_greater_equal_elem(lhs.tensor(), rhs)
    }

    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_lower(lhs.tensor(), rhs.tensor())
    }

    fn lower_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::float_lower_elem(lhs.tensor(), rhs)
    }

    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_lower_equal(lhs.tensor(), rhs.tensor())
    }

    fn lower_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::float_lower_equal_elem(lhs.tensor(), rhs)
    }

    fn argmax(tensor: Self::Primitive, dim: usize) -> IntTensor<B> {
        match tensor {
            TensorPrimitive::Float(tensor) => B::float_argmax(tensor, dim),
            TensorPrimitive::QFloat(tensor) => B::q_argmax(tensor, dim),
        }
    }

    fn argmin(tensor: Self::Primitive, dim: usize) -> IntTensor<B> {
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
    ) -> (Self::Primitive, IntTensor<B>) {
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
    ) -> (Self::Primitive, IntTensor<B>) {
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

    fn clamp(tensor: Self::Primitive, min: Scalar, max: Scalar) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_clamp(tensor, min, max))
            }
            TensorPrimitive::QFloat(tensor) => B::q_clamp(tensor, min, max),
        }
    }

    fn clamp_min(tensor: Self::Primitive, min: Scalar) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_clamp_min(tensor, min))
            }
            TensorPrimitive::QFloat(tensor) => B::q_clamp_min(tensor, min),
        }
    }

    fn clamp_max(tensor: Self::Primitive, max: Scalar) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_clamp_max(tensor, max))
            }
            TensorPrimitive::QFloat(tensor) => B::q_clamp_max(tensor, max),
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
}

impl<B: AutodiffBackend> BasicAutodiffOps<B> for Float {
    type InnerKind = Float;

    fn inner(
        tensor: <Self as TensorKind<B>>::Primitive,
    ) -> <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::inner(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_inner(tensor)),
        }
    }

    fn from_inner(
        inner: <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive,
    ) -> <Self as TensorKind<B>>::Primitive {
        match inner {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::from_inner(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_from_inner(tensor)),
        }
    }
}
