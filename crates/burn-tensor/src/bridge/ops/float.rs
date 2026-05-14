use alloc::vec::Vec;
use burn_backend::{
    AutodiffBackend, Distribution, Scalar, TensorData, TensorMetadata, TensorPrimitive,
    get_device_settings,
    ops::{FloatTensorOps, QTensorOps, TransactionPrimitive},
};
use burn_dispatch::Dispatch;
use burn_std::{DType, ExecutionError, IndexingUpdateOp, Shape, Slice};

use crate::{
    Device, Float,
    bridge::{BasicAutodiffOps, BasicOps, FloatMathOps, Numeric, Ordered, TransactionOp},
    ops::{BoolTensor, IntTensor},
};

macro_rules! q_bin_ops {
    ($lhs:ident, $rhs:ident, $op:ident, $q_op:ident) => {
        match ($lhs, $rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(Dispatch::$op(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                Dispatch::$q_op(lhs, rhs)
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::Float(rhs)) => {
                let dtype = rhs.dtype();
                TensorPrimitive::Float(Dispatch::$op(Dispatch::dequantize(lhs, dtype.into()), rhs))
            }
            (TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs)) => {
                let dtype = lhs.dtype();
                TensorPrimitive::Float(Dispatch::$op(lhs, Dispatch::dequantize(rhs, dtype.into())))
            }
        }
    };
}
impl TransactionOp for Float {
    fn register_transaction(tr: &mut TransactionPrimitive<Dispatch>, tensor: Self::Primitive) {
        tr.register_float(tensor);
    }
}

impl BasicOps for Float {
    fn empty(shape: Shape, device: &Device, dtype: DType) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_empty(shape, &device.dispatch, dtype.into()))
    }

    fn zeros(shape: Shape, device: &Device, dtype: DType) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_zeros(shape, &device.dispatch, dtype.into()))
    }
    fn ones(shape: Shape, device: &Device, dtype: DType) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_ones(shape, &device.dispatch, dtype.into()))
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device, dtype: DType) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_full(
            shape,
            fill_value,
            &device.dispatch,
            dtype.into(),
        ))
    }

    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_reshape(tensor, shape))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_reshape(tensor, shape))
            }
        }
    }

    fn transpose(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_transpose(tensor))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_transpose(tensor))
            }
        }
    }

    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_swap_dims(tensor, dim1, dim2))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_swap_dims(tensor, dim1, dim2))
            }
        }
    }

    fn slice(tensor: Self::Primitive, slices: &[Slice]) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_slice(tensor, slices))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_slice(tensor, slices))
            }
        }
    }

    fn slice_assign(
        tensor: Self::Primitive,
        slices: &[Slice],
        value: Self::Primitive,
    ) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_slice_assign(
            tensor.tensor(),
            slices,
            value.tensor(),
        ))
    }

    fn select(tensor: Self::Primitive, dim: usize, indices: IntTensor) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_select(tensor, dim, indices))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_select(tensor, dim, indices))
            }
        }
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: IntTensor,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        // Select assign is ambiguous for QFloat
        match update {
            IndexingUpdateOp::Add => TensorPrimitive::Float(Dispatch::float_select_add(
                tensor.tensor(),
                dim,
                indices,
                values.tensor(),
            )),
            other => unimplemented!("Unsupported update op {other:?}"),
        }
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: BoolTensor,
        source: Self::Primitive,
    ) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_mask_where(
            tensor.tensor(),
            mask,
            source.tensor(),
        ))
    }

    fn mask_fill(tensor: Self::Primitive, mask: BoolTensor, value: Scalar) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_mask_fill(tensor.tensor(), mask, value))
    }

    fn gather(dim: usize, tensor: Self::Primitive, indices: IntTensor) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_gather(dim, tensor, indices))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_gather(dim, tensor, indices))
            }
        }
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: IntTensor,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => TensorPrimitive::Float(Dispatch::float_scatter_add(
                dim,
                tensor.tensor(),
                indices,
                values.tensor(),
            )),
            other => unimplemented!("Unsupported update op {other:?}"),
        }
    }

    fn scatter_nd(
        data: Self::Primitive,
        indices: IntTensor,
        values: Self::Primitive,
        reduction: IndexingUpdateOp,
    ) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_scatter_nd(
            data.tensor(),
            indices,
            values.tensor(),
            reduction,
        ))
    }

    fn gather_nd(data: Self::Primitive, indices: IntTensor) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_gather_nd(data.tensor(), indices))
    }

    fn device(tensor: &Self::Primitive) -> Device {
        match tensor {
            TensorPrimitive::Float(tensor) => Dispatch::float_device(tensor).into(),
            TensorPrimitive::QFloat(tensor) => Dispatch::q_device(tensor).into(),
        }
    }

    fn to_device(tensor: Self::Primitive, device: &Device) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_to_device(tensor, &device.dispatch))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_to_device(tensor, &device.dispatch))
            }
        }
    }

    async fn into_data_async(tensor: Self::Primitive) -> Result<TensorData, ExecutionError> {
        match tensor {
            TensorPrimitive::Float(tensor) => Dispatch::float_into_data(tensor).await,
            TensorPrimitive::QFloat(tensor) => Dispatch::q_into_data(tensor).await,
        }
    }

    fn from_data(data: TensorData, device: &Device, dtype: DType) -> Self::Primitive {
        if matches!(data.dtype, DType::QFloat(_)) {
            // When the source is QFloat, there is no conversion path possible.
            TensorPrimitive::QFloat(Dispatch::q_from_data(data, &device.dispatch))
        } else if dtype.is_float() {
            TensorPrimitive::Float(Dispatch::float_from_data(
                data.convert_dtype(dtype),
                &device.dispatch,
            ))
        } else {
            panic!("Expected float dtype, got {dtype:?}")
        }
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_repeat_dim(tensor, dim, times))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_repeat_dim(tensor, dim, times))
            }
        }
    }

    fn cat(vectors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        match vectors.first().unwrap() {
            TensorPrimitive::Float(_) => TensorPrimitive::Float(Dispatch::float_cat(
                vectors.into_iter().map(|tensor| tensor.tensor()).collect(),
                dim,
            )),
            TensorPrimitive::QFloat(_) => TensorPrimitive::QFloat(Dispatch::q_cat(
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

    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let lhs = lhs.tensor();
        // TODO: DispatchDevice.settings()
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_equal(lhs, rhs.tensor(), out_dtype)
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_not_equal(lhs, rhs.tensor(), out_dtype)
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_equal_elem(lhs, rhs, out_dtype)
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_not_equal_elem(lhs, rhs, out_dtype)
    }

    fn any(tensor: Self::Primitive) -> BoolTensor {
        let tensor = tensor.tensor();
        let out_dtype =
            get_device_settings::<Dispatch>(&Dispatch::float_device(&tensor)).bool_dtype;
        Dispatch::float_any(tensor, out_dtype)
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> BoolTensor {
        let tensor = tensor.tensor();
        let out_dtype =
            get_device_settings::<Dispatch>(&Dispatch::float_device(&tensor)).bool_dtype;
        Dispatch::float_any_dim(tensor, dim, out_dtype)
    }

    fn all(tensor: Self::Primitive) -> BoolTensor {
        let tensor = tensor.tensor();
        let out_dtype =
            get_device_settings::<Dispatch>(&Dispatch::float_device(&tensor)).bool_dtype;
        Dispatch::float_all(tensor, out_dtype)
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> BoolTensor {
        let tensor = tensor.tensor();
        let out_dtype =
            get_device_settings::<Dispatch>(&Dispatch::float_device(&tensor)).bool_dtype;
        Dispatch::float_all_dim(tensor, dim, out_dtype)
    }

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_permute(tensor, axes))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_permute(tensor, axes))
            }
        }
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_expand(tensor.tensor(), shape))
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_flip(tensor, axes))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_flip(tensor, axes))
            }
        }
    }

    fn unfold(tensor: Self::Primitive, dim: usize, size: usize, step: usize) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_unfold(tensor.tensor(), dim, size, step))
    }
}

impl Numeric for Float {
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_add, q_add)
    }

    fn add_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(Dispatch::float_add_scalar(lhs, rhs))
            }
            TensorPrimitive::QFloat(lhs) => Dispatch::q_add_scalar(lhs, rhs),
        }
    }

    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_sub, q_sub)
    }

    fn sub_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(Dispatch::float_sub_scalar(lhs, rhs))
            }
            TensorPrimitive::QFloat(lhs) => Dispatch::q_sub_scalar(lhs, rhs),
        }
    }

    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_div, q_div)
    }

    fn div_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(Dispatch::float_div_scalar(lhs, rhs))
            }
            TensorPrimitive::QFloat(lhs) => Dispatch::q_div_scalar(lhs, rhs),
        }
    }
    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_remainder(lhs.tensor(), rhs.tensor()))
    }

    fn remainder_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_remainder_scalar(lhs.tensor(), rhs))
    }

    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_mul, q_mul)
    }

    fn mul_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(Dispatch::float_mul_scalar(lhs, rhs))
            }
            TensorPrimitive::QFloat(lhs) => Dispatch::q_mul_scalar(lhs, rhs),
        }
    }
    fn neg(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(Dispatch::float_neg(tensor)),
            TensorPrimitive::QFloat(tensor) => Dispatch::q_neg(tensor),
        }
    }

    fn sum(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(Dispatch::float_sum(tensor)),
            TensorPrimitive::QFloat(tensor) => Dispatch::q_sum(tensor),
        }
    }

    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_sum_dim(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => Dispatch::q_sum_dim(tensor, dim),
        }
    }

    fn prod(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(Dispatch::float_prod(tensor)),
            TensorPrimitive::QFloat(tensor) => Dispatch::q_prod(tensor),
        }
    }

    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_prod_dim(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => Dispatch::q_prod_dim(tensor, dim),
        }
    }

    fn mean(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(Dispatch::float_mean(tensor)),
            TensorPrimitive::QFloat(tensor) => Dispatch::q_mean(tensor),
        }
    }

    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_mean_dim(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => Dispatch::q_mean_dim(tensor, dim),
        }
    }

    fn cumsum(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_cumsum(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => Dispatch::q_cumsum(tensor, dim),
        }
    }

    fn cumprod(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_cumprod(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => Dispatch::q_cumprod(tensor, dim),
        }
    }

    fn abs(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(Dispatch::float_abs(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(Dispatch::q_abs(tensor)),
        }
    }

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        q_bin_ops!(lhs, rhs, float_powf, q_powf)
    }

    fn powi_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(Dispatch::float_powi_scalar(lhs, rhs))
            }
            TensorPrimitive::QFloat(lhs) => Dispatch::q_powi_scalar(lhs, rhs),
        }
    }

    fn random(
        shape: Shape,
        distribution: Distribution,
        device: &Device,
        dtype: DType,
    ) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_random(
            shape,
            distribution,
            &device.dispatch,
            dtype.into(),
        ))
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_sign(tensor.tensor()))
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
                TensorPrimitive::Float(Dispatch::float_matmul(lhs, rhs))
            }
            (lhs, rhs) => Dispatch::q_matmul(lhs, rhs),
        }
    }
}
impl Ordered for Float {
    fn sort(tensor: Self::Primitive, dim: usize, descending: bool) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_sort(tensor, dim, descending))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_sort(tensor, dim, descending))
            }
        }
    }

    fn sort_with_indices(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive, IntTensor) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::float_device(&tensor)).int_dtype;
                let (values, indices) =
                    Dispatch::float_sort_with_indices(tensor, dim, descending, out_dtype);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::q_device(&tensor)).int_dtype;
                let (values, indices) =
                    Dispatch::q_sort_with_indices(tensor, dim, descending, out_dtype);
                (TensorPrimitive::QFloat(values), indices)
            }
        }
    }

    fn argsort(tensor: Self::Primitive, dim: usize, descending: bool) -> IntTensor {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::float_device(&tensor)).int_dtype;
                Dispatch::float_argsort(tensor, dim, descending, out_dtype)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::q_device(&tensor)).int_dtype;
                Dispatch::q_argsort(tensor, dim, descending, out_dtype)
            }
        }
    }

    fn cummin(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_cummin(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => Dispatch::q_cummin(tensor, dim),
        }
    }

    fn cummax(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_cummax(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => Dispatch::q_cummax(tensor, dim),
        }
    }

    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_greater(lhs, rhs.tensor(), out_dtype)
    }

    fn greater_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_greater_elem(lhs, rhs, out_dtype)
    }

    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_greater_equal(lhs, rhs.tensor(), out_dtype)
    }

    fn greater_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_greater_equal_elem(lhs, rhs, out_dtype)
    }

    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_lower(lhs, rhs.tensor(), out_dtype)
    }

    fn lower_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_lower_elem(lhs, rhs, out_dtype)
    }

    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_lower_equal(lhs, rhs.tensor(), out_dtype)
    }

    fn lower_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let lhs = lhs.tensor();
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::float_device(&lhs)).bool_dtype;
        Dispatch::float_lower_equal_elem(lhs, rhs, out_dtype)
    }

    fn argmax(tensor: Self::Primitive, dim: usize) -> IntTensor {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::float_device(&tensor)).int_dtype;
                Dispatch::float_argmax(tensor, dim, out_dtype)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::q_device(&tensor)).int_dtype;
                Dispatch::q_argmax(tensor, dim, out_dtype)
            }
        }
    }

    fn argtopk(tensor: Self::Primitive, dim: usize, k: usize) -> IntTensor {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::float_device(&tensor)).int_dtype;
                Dispatch::float_argtopk(tensor, dim, k, out_dtype)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::q_device(&tensor)).int_dtype;
                Dispatch::q_argtopk(tensor, dim, k, out_dtype)
            }
        }
    }

    fn argmin(tensor: Self::Primitive, dim: usize) -> IntTensor {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::float_device(&tensor)).int_dtype;
                Dispatch::float_argmin(tensor, dim, out_dtype)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::q_device(&tensor)).int_dtype;
                Dispatch::q_argmin(tensor, dim, out_dtype)
            }
        }
    }

    fn max(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(Dispatch::float_max(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(Dispatch::q_max(tensor)),
        }
    }

    fn max_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_max_dim(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_max_dim(tensor, dim))
            }
        }
    }

    fn topk(tensor: Self::Primitive, dim: usize, k: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_topk(tensor, dim, k))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_topk(tensor, dim, k))
            }
        }
    }

    fn max_dim_with_indices(tensor: Self::Primitive, dim: usize) -> (Self::Primitive, IntTensor) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::float_device(&tensor)).int_dtype;
                let (values, indices) =
                    Dispatch::float_max_dim_with_indices(tensor, dim, out_dtype);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::q_device(&tensor)).int_dtype;
                let (values, indices) = Dispatch::q_max_dim_with_indices(tensor, dim, out_dtype);
                (TensorPrimitive::QFloat(values), indices)
            }
        }
    }

    fn min(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(Dispatch::float_min(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(Dispatch::q_min(tensor)),
        }
    }

    fn min_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_min_dim(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_min_dim(tensor, dim))
            }
        }
    }

    fn min_dim_with_indices(tensor: Self::Primitive, dim: usize) -> (Self::Primitive, IntTensor) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::float_device(&tensor)).int_dtype;
                let (values, indices) =
                    Dispatch::float_min_dim_with_indices(tensor, dim, out_dtype);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let out_dtype =
                    get_device_settings::<Dispatch>(&Dispatch::q_device(&tensor)).int_dtype;
                let (values, indices) = Dispatch::q_min_dim_with_indices(tensor, dim, out_dtype);
                (TensorPrimitive::QFloat(values), indices)
            }
        }
    }

    fn clamp(tensor: Self::Primitive, min: Scalar, max: Scalar) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_clamp(tensor, min, max))
            }
            TensorPrimitive::QFloat(tensor) => Dispatch::q_clamp(tensor, min, max),
        }
    }

    fn clamp_min(tensor: Self::Primitive, min: Scalar) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_clamp_min(tensor, min))
            }
            TensorPrimitive::QFloat(tensor) => Dispatch::q_clamp_min(tensor, min),
        }
    }

    fn clamp_max(tensor: Self::Primitive, max: Scalar) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_clamp_max(tensor, max))
            }
            TensorPrimitive::QFloat(tensor) => Dispatch::q_clamp_max(tensor, max),
        }
    }

    fn max_abs(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_max_abs(tensor))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(Dispatch::q_max_abs(tensor)),
        }
    }

    fn max_abs_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(Dispatch::float_max_abs_dim(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_max_abs_dim(tensor, dim))
            }
        }
    }
}

impl FloatMathOps for Float {
    fn square(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_powi_scalar(tensor.tensor(), 2.into()))
    }
    fn sqrt(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_sqrt(tensor.tensor()))
    }
    fn cos(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_cos(tensor.tensor()))
    }

    fn sin(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_sin(tensor.tensor()))
    }

    fn tan(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_tan(tensor.tensor()))
    }

    fn cosh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_cosh(tensor.tensor()))
    }

    fn sinh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_sinh(tensor.tensor()))
    }

    fn tanh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_tanh(tensor.tensor()))
    }

    fn acos(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_acos(tensor.tensor()))
    }

    fn acosh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_acosh(tensor.tensor()))
    }

    fn asin(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_asin(tensor.tensor()))
    }

    fn asinh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_asinh(tensor.tensor()))
    }

    fn atan(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_atan(tensor.tensor()))
    }

    fn atanh(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_atanh(tensor.tensor()))
    }

    fn atan2(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_atan2(lhs.tensor(), rhs.tensor()))
    }

    fn exp(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_exp(tensor.tensor()))
    }

    fn log(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_log(tensor.tensor()))
    }

    fn log1p(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(Dispatch::float_log1p(tensor.tensor()))
    }
}

impl BasicAutodiffOps for Float {
    fn inner(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(Dispatch::inner(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(Dispatch::q_inner(tensor)),
        }
    }

    fn from_inner(inner: Self::Primitive) -> Self::Primitive {
        match inner {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(Dispatch::from_inner(tensor)),
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(Dispatch::q_from_inner(tensor))
            }
        }
    }
}
