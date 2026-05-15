use alloc::vec::Vec;
use burn_backend::{
    AutodiffBackend, Distribution, Scalar, TensorData, TensorMetadata, TensorPrimitive,
    ops::{FloatTensorOps, QTensorOps, TransactionPrimitive},
};
use burn_dispatch::Dispatch;
use burn_std::{DType, ExecutionError, IndexingUpdateOp, Shape, Slice};

use crate::{
    Device, Float,
    bridge::{BasicAutodiffOps, BasicOps, FloatMathOps, Numeric, Ordered, TransactionOp},
    ops::BridgeTensor,
};

macro_rules! q_bin_ops {
    ($lhs:ident, $rhs:ident, $op:ident, $q_op:ident) => {
        match ($lhs, $rhs) {
            (BridgeTensor::Float(lhs), BridgeTensor::Float(rhs)) => {
                BridgeTensor::Float(Dispatch::$op(lhs, rhs))
            }
            (BridgeTensor::QFloat(lhs), BridgeTensor::QFloat(rhs)) => {
                match Dispatch::$q_op(lhs, rhs) {
                    TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                    TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
                }
            }
            (BridgeTensor::QFloat(lhs), BridgeTensor::Float(rhs)) => {
                let dtype = rhs.dtype();
                BridgeTensor::Float(Dispatch::$op(Dispatch::dequantize(lhs, dtype.into()), rhs))
            }
            (BridgeTensor::Float(lhs), BridgeTensor::QFloat(rhs)) => {
                let dtype = lhs.dtype();
                BridgeTensor::Float(Dispatch::$op(lhs, Dispatch::dequantize(rhs, dtype.into())))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    };
}
impl TransactionOp for Float {
    fn register_transaction(tr: &mut TransactionPrimitive<Dispatch>, tensor: BridgeTensor) {
        match tensor {
            BridgeTensor::Float(tensor) => tr.register_float(TensorPrimitive::Float(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}

impl BasicOps for Float {
    fn empty(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_empty(shape, &device.dispatch, dtype.into()))
    }

    fn zeros(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_zeros(shape, &device.dispatch, dtype.into()))
    }
    fn ones(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_ones(shape, &device.dispatch, dtype.into()))
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_full(
            shape,
            fill_value,
            &device.dispatch,
            dtype.into(),
        ))
    }

    fn reshape(tensor: BridgeTensor, shape: Shape) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_reshape(tensor, shape))
            }
            BridgeTensor::QFloat(tensor) => {
                BridgeTensor::QFloat(Dispatch::q_reshape(tensor, shape))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn transpose(tensor: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_transpose(tensor)),
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_transpose(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn swap_dims(tensor: BridgeTensor, dim1: usize, dim2: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_swap_dims(tensor, dim1, dim2))
            }
            BridgeTensor::QFloat(tensor) => {
                BridgeTensor::QFloat(Dispatch::q_swap_dims(tensor, dim1, dim2))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn slice(tensor: BridgeTensor, slices: &[Slice]) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_slice(tensor, slices))
            }
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_slice(tensor, slices)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn slice_assign(tensor: BridgeTensor, slices: &[Slice], value: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_slice_assign(
            tensor.into_float(),
            slices,
            value.into_float(),
        ))
    }

    fn select(tensor: BridgeTensor, dim: usize, indices: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_select(tensor, dim, indices.into()))
            }
            BridgeTensor::QFloat(tensor) => {
                BridgeTensor::QFloat(Dispatch::q_select(tensor, dim, indices.into()))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn select_assign(
        tensor: BridgeTensor,
        dim: usize,
        indices: BridgeTensor,
        values: BridgeTensor,
        update: IndexingUpdateOp,
    ) -> BridgeTensor {
        // Select assign is ambiguous for QFloat
        match update {
            IndexingUpdateOp::Add => BridgeTensor::Float(Dispatch::float_select_add(
                tensor.into_float(),
                dim,
                indices.into(),
                values.into_float(),
            )),
            other => unimplemented!("Unsupported update op {other:?}"),
        }
    }

    fn mask_where(tensor: BridgeTensor, mask: BridgeTensor, source: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_mask_where(
            tensor.into_float(),
            mask.into(),
            source.into_float(),
        ))
    }

    fn mask_fill(tensor: BridgeTensor, mask: BridgeTensor, value: Scalar) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_mask_fill(
            tensor.into_float(),
            mask.into(),
            value,
        ))
    }

    fn gather(dim: usize, tensor: BridgeTensor, indices: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_gather(dim, tensor, indices.into()))
            }
            BridgeTensor::QFloat(tensor) => {
                BridgeTensor::QFloat(Dispatch::q_gather(dim, tensor, indices.into()))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn scatter(
        dim: usize,
        tensor: BridgeTensor,
        indices: BridgeTensor,
        values: BridgeTensor,
        update: IndexingUpdateOp,
    ) -> BridgeTensor {
        match update {
            IndexingUpdateOp::Add => BridgeTensor::Float(Dispatch::float_scatter_add(
                dim,
                tensor.into_float(),
                indices.into(),
                values.into_float(),
            )),
            other => unimplemented!("Unsupported update op {other:?}"),
        }
    }

    fn scatter_nd(
        data: BridgeTensor,
        indices: BridgeTensor,
        values: BridgeTensor,
        reduction: IndexingUpdateOp,
    ) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_scatter_nd(
            data.into_float(),
            indices.into(),
            values.into_float(),
            reduction,
        ))
    }

    fn gather_nd(data: BridgeTensor, indices: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_gather_nd(data.into_float(), indices.into()))
    }

    fn device(tensor: &BridgeTensor) -> Device {
        match tensor {
            BridgeTensor::Float(tensor) => Dispatch::float_device(tensor).into(),
            BridgeTensor::QFloat(tensor) => Dispatch::q_device(tensor).into(),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn to_device(tensor: BridgeTensor, device: &Device) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_to_device(tensor, &device.dispatch))
            }
            BridgeTensor::QFloat(tensor) => {
                BridgeTensor::QFloat(Dispatch::q_to_device(tensor, &device.dispatch))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    async fn into_data_async(tensor: BridgeTensor) -> Result<TensorData, ExecutionError> {
        match tensor {
            BridgeTensor::Float(tensor) => Dispatch::float_into_data(tensor).await,
            BridgeTensor::QFloat(tensor) => Dispatch::q_into_data(tensor).await,
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn from_data(data: TensorData, device: &Device, dtype: DType) -> BridgeTensor {
        if matches!(data.dtype, DType::QFloat(_)) {
            // When the source is QFloat, there is no conversion path possible.
            BridgeTensor::QFloat(Dispatch::q_from_data(data, &device.dispatch))
        } else if dtype.is_float() {
            BridgeTensor::Float(Dispatch::float_from_data(
                data.convert_dtype(dtype),
                &device.dispatch,
            ))
        } else {
            panic!("Expected float dtype, got {dtype:?}")
        }
    }

    fn repeat_dim(tensor: BridgeTensor, dim: usize, times: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_repeat_dim(tensor, dim, times))
            }
            BridgeTensor::QFloat(tensor) => {
                BridgeTensor::QFloat(Dispatch::q_repeat_dim(tensor, dim, times))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cat(vectors: Vec<BridgeTensor>, dim: usize) -> BridgeTensor {
        match vectors.first().unwrap() {
            BridgeTensor::Float(_) => BridgeTensor::Float(Dispatch::float_cat(
                BridgeTensor::into_dispatch_vec(vectors),
                dim,
            )),
            BridgeTensor::QFloat(_) => BridgeTensor::QFloat(Dispatch::q_cat(
                BridgeTensor::into_dispatch_vec(vectors),
                dim,
            )),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_equal(lhs, rhs.into_float(), out_dtype))
    }

    fn not_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_not_equal(lhs, rhs.into_float(), out_dtype))
    }

    fn equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_equal_elem(lhs, rhs, out_dtype))
    }

    fn not_equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_not_equal_elem(lhs, rhs, out_dtype))
    }

    fn any(tensor: BridgeTensor) -> BridgeTensor {
        let tensor = tensor.into_float();
        let out_dtype = Dispatch::float_device(&tensor).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_any(tensor, out_dtype))
    }

    fn any_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let tensor = tensor.into_float();
        let out_dtype = Dispatch::float_device(&tensor).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_any_dim(tensor, dim, out_dtype))
    }

    fn all(tensor: BridgeTensor) -> BridgeTensor {
        let tensor = tensor.into_float();
        let out_dtype = Dispatch::float_device(&tensor).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_all(tensor, out_dtype))
    }

    fn all_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let tensor = tensor.into_float();
        let out_dtype = Dispatch::float_device(&tensor).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_all_dim(tensor, dim, out_dtype))
    }

    fn permute(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_permute(tensor, axes))
            }
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_permute(tensor, axes)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn expand(tensor: BridgeTensor, shape: Shape) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_expand(tensor.into_float(), shape))
    }

    fn flip(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_flip(tensor, axes)),
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_flip(tensor, axes)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn unfold(tensor: BridgeTensor, dim: usize, size: usize, step: usize) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_unfold(tensor.into_float(), dim, size, step))
    }
}

impl Numeric for Float {
    fn add(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        q_bin_ops!(lhs, rhs, float_add, q_add)
    }

    fn add_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        match lhs {
            BridgeTensor::Float(lhs) => BridgeTensor::Float(Dispatch::float_add_scalar(lhs, rhs)),
            BridgeTensor::QFloat(lhs) => match Dispatch::q_add_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sub(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        q_bin_ops!(lhs, rhs, float_sub, q_sub)
    }

    fn sub_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        match lhs {
            BridgeTensor::Float(lhs) => BridgeTensor::Float(Dispatch::float_sub_scalar(lhs, rhs)),
            BridgeTensor::QFloat(lhs) => match Dispatch::q_sub_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn div(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        q_bin_ops!(lhs, rhs, float_div, q_div)
    }

    fn div_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        match lhs {
            BridgeTensor::Float(lhs) => BridgeTensor::Float(Dispatch::float_div_scalar(lhs, rhs)),
            BridgeTensor::QFloat(lhs) => match Dispatch::q_div_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }
    fn remainder(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_remainder(
            lhs.into_float(),
            rhs.into_float(),
        ))
    }

    fn remainder_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_remainder_scalar(lhs.into_float(), rhs))
    }

    fn mul(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        q_bin_ops!(lhs, rhs, float_mul, q_mul)
    }

    fn mul_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        match lhs {
            BridgeTensor::Float(lhs) => BridgeTensor::Float(Dispatch::float_mul_scalar(lhs, rhs)),
            BridgeTensor::QFloat(lhs) => match Dispatch::q_mul_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }
    fn neg(tensor: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_neg(tensor)),
            BridgeTensor::QFloat(tensor) => match Dispatch::q_neg(tensor) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sum(tensor: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_sum(tensor)),
            BridgeTensor::QFloat(tensor) => match Dispatch::q_sum(tensor) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sum_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_sum_dim(tensor, dim))
            }
            BridgeTensor::QFloat(tensor) => match Dispatch::q_sum_dim(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn prod(tensor: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_prod(tensor)),
            BridgeTensor::QFloat(tensor) => match Dispatch::q_prod(tensor) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn prod_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_prod_dim(tensor, dim))
            }
            BridgeTensor::QFloat(tensor) => match Dispatch::q_prod_dim(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn mean(tensor: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_mean(tensor)),
            BridgeTensor::QFloat(tensor) => match Dispatch::q_mean(tensor) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn mean_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_mean_dim(tensor, dim))
            }
            BridgeTensor::QFloat(tensor) => match Dispatch::q_mean_dim(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cumsum(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_cumsum(tensor, dim)),
            BridgeTensor::QFloat(tensor) => match Dispatch::q_cumsum(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cumprod(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_cumprod(tensor, dim))
            }
            BridgeTensor::QFloat(tensor) => match Dispatch::q_cumprod(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn abs(tensor: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_abs(tensor)),
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_abs(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn powi(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        q_bin_ops!(lhs, rhs, float_powf, q_powf)
    }

    fn powi_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        match lhs {
            BridgeTensor::Float(lhs) => BridgeTensor::Float(Dispatch::float_powi_scalar(lhs, rhs)),
            BridgeTensor::QFloat(lhs) => match Dispatch::q_powi_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn random(
        shape: Shape,
        distribution: Distribution,
        device: &Device,
        dtype: DType,
    ) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_random(
            shape,
            distribution,
            &device.dispatch,
            dtype.into(),
        ))
    }

    fn sign(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_sign(tensor.into_float()))
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// # Panics
    ///
    /// If the two tensors don't have a compatible shape.
    fn matmul(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        match (lhs, rhs) {
            (BridgeTensor::Float(lhs), BridgeTensor::Float(rhs)) => {
                BridgeTensor::Float(Dispatch::float_matmul(lhs, rhs))
            }
            (BridgeTensor::Float(lhs), BridgeTensor::QFloat(rhs)) => {
                match Dispatch::q_matmul(TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs))
                {
                    TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                    TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
                }
            }
            (BridgeTensor::QFloat(lhs), BridgeTensor::Float(rhs)) => {
                match Dispatch::q_matmul(TensorPrimitive::QFloat(lhs), TensorPrimitive::Float(rhs))
                {
                    TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                    TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
                }
            }
            (BridgeTensor::QFloat(lhs), BridgeTensor::QFloat(rhs)) => {
                match Dispatch::q_matmul(TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs))
                {
                    TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                    TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
                }
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }
}
impl Ordered for Float {
    fn sort(tensor: BridgeTensor, dim: usize, descending: bool) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_sort(tensor, dim, descending))
            }
            BridgeTensor::QFloat(tensor) => {
                BridgeTensor::QFloat(Dispatch::q_sort(tensor, dim, descending))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sort_with_indices(
        tensor: BridgeTensor,
        dim: usize,
        descending: bool,
    ) -> (BridgeTensor, BridgeTensor) {
        match tensor {
            BridgeTensor::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                let (values, indices) =
                    Dispatch::float_sort_with_indices(tensor, dim, descending, out_dtype);
                (BridgeTensor::Float(values), BridgeTensor::Int(indices))
            }
            BridgeTensor::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                let (values, indices) =
                    Dispatch::q_sort_with_indices(tensor, dim, descending, out_dtype);
                (BridgeTensor::QFloat(values), BridgeTensor::Int(indices))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn argsort(tensor: BridgeTensor, dim: usize, descending: bool) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                BridgeTensor::Int(Dispatch::float_argsort(tensor, dim, descending, out_dtype))
            }
            BridgeTensor::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                BridgeTensor::Int(Dispatch::q_argsort(tensor, dim, descending, out_dtype))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cummin(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_cummin(tensor, dim)),
            BridgeTensor::QFloat(tensor) => match Dispatch::q_cummin(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cummax(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_cummax(tensor, dim)),
            BridgeTensor::QFloat(tensor) => match Dispatch::q_cummax(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn greater(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_greater(lhs, rhs.into_float(), out_dtype))
    }

    fn greater_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_greater_elem(lhs, rhs, out_dtype))
    }

    fn greater_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_greater_equal(
            lhs,
            rhs.into_float(),
            out_dtype,
        ))
    }

    fn greater_equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_greater_equal_elem(lhs, rhs, out_dtype))
    }

    fn lower(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_lower(lhs, rhs.into_float(), out_dtype))
    }

    fn lower_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_lower_elem(lhs, rhs, out_dtype))
    }

    fn lower_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_lower_equal(
            lhs,
            rhs.into_float(),
            out_dtype,
        ))
    }

    fn lower_equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        BridgeTensor::Bool(Dispatch::float_lower_equal_elem(lhs, rhs, out_dtype))
    }

    fn argmax(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                BridgeTensor::Int(Dispatch::float_argmax(tensor, dim, out_dtype))
            }
            BridgeTensor::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                BridgeTensor::Int(Dispatch::q_argmax(tensor, dim, out_dtype))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn argtopk(tensor: BridgeTensor, dim: usize, k: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                BridgeTensor::Int(Dispatch::float_argtopk(tensor, dim, k, out_dtype))
            }
            BridgeTensor::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                BridgeTensor::Int(Dispatch::q_argtopk(tensor, dim, k, out_dtype))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn argmin(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                BridgeTensor::Int(Dispatch::float_argmin(tensor, dim, out_dtype))
            }
            BridgeTensor::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                BridgeTensor::Int(Dispatch::q_argmin(tensor, dim, out_dtype))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max(tensor: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_max(tensor)),
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_max(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_max_dim(tensor, dim))
            }
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_max_dim(tensor, dim)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn topk(tensor: BridgeTensor, dim: usize, k: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_topk(tensor, dim, k))
            }
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_topk(tensor, dim, k)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_dim_with_indices(tensor: BridgeTensor, dim: usize) -> (BridgeTensor, BridgeTensor) {
        match tensor {
            BridgeTensor::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                let (values, indices) =
                    Dispatch::float_max_dim_with_indices(tensor, dim, out_dtype);
                (BridgeTensor::Float(values), BridgeTensor::Int(indices))
            }
            BridgeTensor::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                let (values, indices) = Dispatch::q_max_dim_with_indices(tensor, dim, out_dtype);
                (BridgeTensor::QFloat(values), BridgeTensor::Int(indices))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn min(tensor: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_min(tensor)),
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_min(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn min_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_min_dim(tensor, dim))
            }
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_min_dim(tensor, dim)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn min_dim_with_indices(tensor: BridgeTensor, dim: usize) -> (BridgeTensor, BridgeTensor) {
        match tensor {
            BridgeTensor::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                let (values, indices) =
                    Dispatch::float_min_dim_with_indices(tensor, dim, out_dtype);
                (BridgeTensor::Float(values), BridgeTensor::Int(indices))
            }
            BridgeTensor::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                let (values, indices) = Dispatch::q_min_dim_with_indices(tensor, dim, out_dtype);
                (BridgeTensor::QFloat(values), BridgeTensor::Int(indices))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn clamp(tensor: BridgeTensor, min: Scalar, max: Scalar) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_clamp(tensor, min, max))
            }
            BridgeTensor::QFloat(tensor) => match Dispatch::q_clamp(tensor, min, max) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn clamp_min(tensor: BridgeTensor, min: Scalar) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_clamp_min(tensor, min))
            }
            BridgeTensor::QFloat(tensor) => match Dispatch::q_clamp_min(tensor, min) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn clamp_max(tensor: BridgeTensor, max: Scalar) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_clamp_max(tensor, max))
            }
            BridgeTensor::QFloat(tensor) => match Dispatch::q_clamp_max(tensor, max) {
                TensorPrimitive::Float(out) => BridgeTensor::Float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_abs(tensor: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::float_max_abs(tensor)),
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_max_abs(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_abs_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => {
                BridgeTensor::Float(Dispatch::float_max_abs_dim(tensor, dim))
            }
            BridgeTensor::QFloat(tensor) => {
                BridgeTensor::QFloat(Dispatch::q_max_abs_dim(tensor, dim))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }
}

impl FloatMathOps for Float {
    fn square(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_powi_scalar(tensor.into_float(), 2.into()))
    }
    fn sqrt(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_sqrt(tensor.into_float()))
    }
    fn cos(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_cos(tensor.into_float()))
    }

    fn sin(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_sin(tensor.into_float()))
    }

    fn tan(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_tan(tensor.into_float()))
    }

    fn cosh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_cosh(tensor.into_float()))
    }

    fn sinh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_sinh(tensor.into_float()))
    }

    fn tanh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_tanh(tensor.into_float()))
    }

    fn acos(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_acos(tensor.into_float()))
    }

    fn acosh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_acosh(tensor.into_float()))
    }

    fn asin(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_asin(tensor.into_float()))
    }

    fn asinh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_asinh(tensor.into_float()))
    }

    fn atan(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_atan(tensor.into_float()))
    }

    fn atanh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_atanh(tensor.into_float()))
    }

    fn atan2(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_atan2(lhs.into_float(), rhs.into_float()))
    }

    fn exp(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_exp(tensor.into_float()))
    }

    fn log(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_log(tensor.into_float()))
    }

    fn log1p(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Float(Dispatch::float_log1p(tensor.into_float()))
    }
}

impl BasicAutodiffOps for Float {
    fn inner(tensor: BridgeTensor) -> BridgeTensor {
        match tensor {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::inner(tensor)),
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_inner(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn from_inner(inner: BridgeTensor) -> BridgeTensor {
        match inner {
            BridgeTensor::Float(tensor) => BridgeTensor::Float(Dispatch::from_inner(tensor)),
            BridgeTensor::QFloat(tensor) => BridgeTensor::QFloat(Dispatch::q_from_inner(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}
