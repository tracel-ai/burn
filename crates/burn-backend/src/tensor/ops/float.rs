use alloc::vec::Vec;
use burn_std::{DType, Shape, Slice};

use crate::{
    AutodiffBackend, Backend, Distribution, ExecutionError, Scalar, TensorData, TensorMetadata,
    TensorPrimitive, get_device_settings,
    ops::TransactionPrimitive,
    tensor::{
        BasicAutodiffOps, BasicOps, Device, Float, IndexingUpdateOp, IntTensor, Numeric, Ordered,
        TensorKind, TransactionOp,
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
                let dtype = rhs.dtype();
                TensorPrimitive::Float(B::$op(B::dequantize(lhs, dtype.into()), rhs))
            }
            (TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs)) => {
                let dtype = lhs.dtype();
                TensorPrimitive::Float(B::$op(lhs, B::dequantize(rhs, dtype.into())))
            }
        }
    };
}
impl<B: Backend> TransactionOp<B> for Float {
    fn register_transaction(tr: &mut TransactionPrimitive<B>, tensor: Self::Primitive) {
        tr.register_float(tensor);
    }
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
            _ => unimplemented!(),
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
            _ => unimplemented!(),
        }
    }

    fn scatter_nd(
        data: Self::Primitive,
        indices: IntTensor<B>,
        values: Self::Primitive,
        reduction: IndexingUpdateOp,
    ) -> Self::Primitive {
        TensorPrimitive::Float(B::float_scatter_nd(
            data.tensor(),
            indices,
            values.tensor(),
            reduction,
        ))
    }

    fn gather_nd(data: Self::Primitive, indices: IntTensor<B>) -> Self::Primitive {
        TensorPrimitive::Float(B::float_gather_nd(data.tensor(), indices))
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

    fn from_data(data: TensorData, device: &Device<B>, dtype: DType) -> Self::Primitive {
        if matches!(data.dtype, DType::QFloat(_)) {
            // When the source is QFloat, there is no conversion path possible.
            TensorPrimitive::QFloat(B::q_from_data(data, device))
        } else if dtype.is_float() {
            TensorPrimitive::Float(B::float_from_data(data.convert_dtype(dtype), device))
        } else {
            panic!("Expected float dtype, got {dtype:?}")
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
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_equal(lhs, rhs.tensor(), out_dtype)
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_not_equal(lhs, rhs.tensor(), out_dtype)
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_equal_elem(lhs, rhs, out_dtype)
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_not_equal_elem(lhs, rhs, out_dtype)
    }

    fn any(tensor: Self::Primitive) -> B::BoolTensorPrimitive {
        let tensor = tensor.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&tensor)).bool_dtype;
        B::float_any(tensor, out_dtype)
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> B::BoolTensorPrimitive {
        let tensor = tensor.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&tensor)).bool_dtype;
        B::float_any_dim(tensor, dim, out_dtype)
    }

    fn all(tensor: Self::Primitive) -> B::BoolTensorPrimitive {
        let tensor = tensor.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&tensor)).bool_dtype;
        B::float_all(tensor, out_dtype)
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> B::BoolTensorPrimitive {
        let tensor = tensor.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&tensor)).bool_dtype;
        B::float_all_dim(tensor, dim, out_dtype)
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

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_powf, q_powf)
    }

    fn powi_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => TensorPrimitive::Float(B::float_powi_scalar(lhs, rhs)),
            TensorPrimitive::QFloat(lhs) => B::q_powi_scalar(lhs, rhs),
        }
    }

    fn random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<B>,
        dtype: DType,
    ) -> Self::Primitive {
        TensorPrimitive::Float(B::float_random(shape, distribution, device, dtype.into()))
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
                let out_dtype = get_device_settings::<B>(&B::float_device(&tensor)).int_dtype;
                let (values, indices) =
                    B::float_sort_with_indices(tensor, dim, descending, out_dtype);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::q_device(&tensor)).int_dtype;
                let (values, indices) = B::q_sort_with_indices(tensor, dim, descending, out_dtype);
                (TensorPrimitive::QFloat(values), indices)
            }
        }
    }

    fn argsort(tensor: Self::Primitive, dim: usize, descending: bool) -> IntTensor<B> {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::float_device(&tensor)).int_dtype;
                B::float_argsort(tensor, dim, descending, out_dtype)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::q_device(&tensor)).int_dtype;
                B::q_argsort(tensor, dim, descending, out_dtype)
            }
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
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_greater(lhs, rhs.tensor(), out_dtype)
    }

    fn greater_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_greater_elem(lhs, rhs, out_dtype)
    }

    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_greater_equal(lhs, rhs.tensor(), out_dtype)
    }

    fn greater_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_greater_equal_elem(lhs, rhs, out_dtype)
    }

    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_lower(lhs, rhs.tensor(), out_dtype)
    }

    fn lower_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_lower_elem(lhs, rhs, out_dtype)
    }

    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_lower_equal(lhs, rhs.tensor(), out_dtype)
    }

    fn lower_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<B>(&B::float_device(&lhs)).bool_dtype;
        B::float_lower_equal_elem(lhs, rhs, out_dtype)
    }

    fn argmax(tensor: Self::Primitive, dim: usize) -> IntTensor<B> {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::float_device(&tensor)).int_dtype;
                B::float_argmax(tensor, dim, out_dtype)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::q_device(&tensor)).int_dtype;
                B::q_argmax(tensor, dim, out_dtype)
            }
        }
    }

    fn argtopk(tensor: Self::Primitive, dim: usize, k: usize) -> IntTensor<B> {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::float_device(&tensor)).int_dtype;
                B::float_argtopk(tensor, dim, k, out_dtype)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::q_device(&tensor)).int_dtype;
                B::q_argtopk(tensor, dim, k, out_dtype)
            }
        }
    }

    fn argmin(tensor: Self::Primitive, dim: usize) -> IntTensor<B> {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::float_device(&tensor)).int_dtype;
                B::float_argmin(tensor, dim, out_dtype)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::q_device(&tensor)).int_dtype;
                B::q_argmin(tensor, dim, out_dtype)
            }
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

    fn topk(tensor: Self::Primitive, dim: usize, k: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_topk(tensor, dim, k)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_topk(tensor, dim, k)),
        }
    }

    fn max_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, IntTensor<B>) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::float_device(&tensor)).int_dtype;
                let (values, indices) = B::float_max_dim_with_indices(tensor, dim, out_dtype);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::q_device(&tensor)).int_dtype;
                let (values, indices) = B::q_max_dim_with_indices(tensor, dim, out_dtype);
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
                let out_dtype = get_device_settings::<B>(&B::float_device(&tensor)).int_dtype;
                let (values, indices) = B::float_min_dim_with_indices(tensor, dim, out_dtype);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype = get_device_settings::<B>(&B::q_device(&tensor)).int_dtype;
                let (values, indices) = B::q_min_dim_with_indices(tensor, dim, out_dtype);
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

/// Trait that lists some floating-point mathematical operations are common to all float-like dtypes.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by the
#[cfg_attr(doc, doc = crate::doc_tensor!())]
#[cfg_attr(not(doc), doc = "`Tensor`")]
/// struct.
pub trait FloatMathOps<B: Backend>: Numeric<B> {
    /// Applies element wise square operation
    ///
    #[cfg_attr(doc, doc = "$y_i = x^{2}$")]
    #[cfg_attr(not(doc), doc = "`y = x^2`")]
    fn square(tensor: Self::Primitive) -> Self::Primitive;

    /// Applies element wise exponential operation.
    ///
    #[cfg_attr(doc, doc = "$y_i = e^{x_i}$")]
    #[cfg_attr(not(doc), doc = "`y = e^x`")]
    fn exp(tensor: Self::Primitive) -> Self::Primitive;

    /// Applies the natural logarithm of one plus the input tensor, element-wise.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \log_e\(x_i + 1\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = log(x_i + 1)`")]
    fn log1p(tensor: Self::Primitive) -> Self::Primitive;

    /// Applies element wise natural log operation *ln*.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \log_e\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = log(x_i)`")]
    fn log(tensor: Self::Primitive) -> Self::Primitive;

    /// Applies element wise root square operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \sqrt{x_i}$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = sqrt(x_i)`")]
    fn sqrt(tensor: Self::Primitive) -> Self::Primitive;
    /// Returns a new tensor with cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the cosine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("cos"))]
    #[cfg_attr(not(doc), doc = "`Tensor::cos`")]
    /// function, which is more high-level and designed for public use.
    fn cos(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the sine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("sin"))]
    #[cfg_attr(not(doc), doc = "`Tensor::sin`")]
    /// function, which is more high-level and designed for public use.
    fn sin(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the tangent of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("tan"))]
    #[cfg_attr(not(doc), doc = "`Tensor::tan`")]
    /// function, which is more high-level and designed for public use.
    fn tan(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with hyperbolic cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the hyperbolic cosine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("cosh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::cosh`")]
    /// function, which is more high-level and designed for public use.
    fn cosh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with hyperbolic sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the hyperbolic sine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("sinh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::sinh`")]
    /// function, which is more high-level and designed for public use.
    fn sinh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with hyperbolic tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the hyperbolic tangent of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("tanh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::tanh`")]
    /// function, which is more high-level and designed for public use.
    fn tanh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse cosine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("acos"))]
    #[cfg_attr(not(doc), doc = "`Tensor::acos`")]
    /// function, which is more high-level and designed for public use.
    fn acos(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse hyperbolic cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse hyperbolic cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse hyperbolic cosine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("acosh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::acosh`")]
    /// function, which is more high-level and designed for public use.
    fn acosh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse sine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("asin"))]
    #[cfg_attr(not(doc), doc = "`Tensor::asin`")]
    /// function, which is more high-level and designed for public use.
    fn asin(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse hyperbolic sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse hyperbolic sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse hyperbolic sine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("asinh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::asinh`")]
    /// function, which is more high-level and designed for public use.
    fn asinh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse tangent of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("atan"))]
    #[cfg_attr(not(doc), doc = "`Tensor::atan`")]
    /// function, which is more high-level and designed for public use.
    fn atan(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse hyperbolic tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse hyperbolic tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse hyperbolic tangent of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("atanh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::atanh`")]
    /// function, which is more high-level and designed for public use.
    fn atanh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a tensor with the four-quadrant inverse tangent values of `y` and `x`.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The tensor with y coordinates.
    /// * `rhs` - The tensor with x coordinates.
    ///
    /// # Returns
    ///
    /// A tensor with the four-quadrant inverse tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the four-quadrant inverse tangent of two tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("atan2"))]
    #[cfg_attr(not(doc), doc = "`Tensor::atan2`")]
    /// function, which is more high-level and designed for public use.
    fn atan2(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;
}

impl<B: Backend> FloatMathOps<B> for Float {
    fn square(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_powi_scalar(tensor.tensor(), 2.into()))
    }
    fn sqrt(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_sqrt(tensor.tensor()))
    }
    fn cos(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_cos(tensor.tensor()))
    }

    fn sin(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_sin(tensor.tensor()))
    }

    fn tan(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_tan(tensor.tensor()))
    }

    fn cosh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_cosh(tensor.tensor()))
    }

    fn sinh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_sinh(tensor.tensor()))
    }

    fn tanh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_tanh(tensor.tensor()))
    }

    fn acos(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_acos(tensor.tensor()))
    }

    fn acosh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_acosh(tensor.tensor()))
    }

    fn asin(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_asin(tensor.tensor()))
    }

    fn asinh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_asinh(tensor.tensor()))
    }

    fn atan(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_atan(tensor.tensor()))
    }

    fn atanh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_atanh(tensor.tensor()))
    }

    fn atan2(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_atan2(lhs.tensor(), rhs.tensor()))
    }

    fn exp(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_exp(tensor.tensor()))
    }

    fn log(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_log(tensor.tensor()))
    }

    fn log1p(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_log1p(tensor.tensor()))
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
