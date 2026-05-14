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
    ops::PrimitiveKind,
};

macro_rules! q_bin_ops {
    ($lhs:ident, $rhs:ident, $op:ident, $q_op:ident) => {
        match ($lhs, $rhs) {
            (PrimitiveKind::Float(lhs), PrimitiveKind::Float(rhs)) => {
                PrimitiveKind::Float(Dispatch::$op(lhs, rhs))
            }
            (PrimitiveKind::QFloat(lhs), PrimitiveKind::QFloat(rhs)) => {
                match Dispatch::$q_op(lhs, rhs) {
                    TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                    TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
                }
            }
            (PrimitiveKind::QFloat(lhs), PrimitiveKind::Float(rhs)) => {
                let dtype = rhs.dtype();
                PrimitiveKind::Float(Dispatch::$op(Dispatch::dequantize(lhs, dtype.into()), rhs))
            }
            (PrimitiveKind::Float(lhs), PrimitiveKind::QFloat(rhs)) => {
                let dtype = lhs.dtype();
                PrimitiveKind::Float(Dispatch::$op(lhs, Dispatch::dequantize(rhs, dtype.into())))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    };
}
impl TransactionOp for Float {
    fn register_transaction(tr: &mut TransactionPrimitive<Dispatch>, tensor: PrimitiveKind) {
        match tensor {
            PrimitiveKind::Float(tensor) => tr.register_float(TensorPrimitive::Float(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}

impl BasicOps for Float {
    fn empty(shape: Shape, device: &Device, dtype: DType) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_empty(shape, &device.dispatch, dtype.into()))
    }

    fn zeros(shape: Shape, device: &Device, dtype: DType) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_zeros(shape, &device.dispatch, dtype.into()))
    }
    fn ones(shape: Shape, device: &Device, dtype: DType) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_ones(shape, &device.dispatch, dtype.into()))
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device, dtype: DType) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_full(
            shape,
            fill_value,
            &device.dispatch,
            dtype.into(),
        ))
    }

    fn reshape(tensor: PrimitiveKind, shape: Shape) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_reshape(tensor, shape))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_reshape(tensor, shape))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn transpose(tensor: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => PrimitiveKind::Float(Dispatch::float_transpose(tensor)),
            PrimitiveKind::QFloat(tensor) => PrimitiveKind::QFloat(Dispatch::q_transpose(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn swap_dims(tensor: PrimitiveKind, dim1: usize, dim2: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_swap_dims(tensor, dim1, dim2))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_swap_dims(tensor, dim1, dim2))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn slice(tensor: PrimitiveKind, slices: &[Slice]) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_slice(tensor, slices))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_slice(tensor, slices))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn slice_assign(
        tensor: PrimitiveKind,
        slices: &[Slice],
        value: PrimitiveKind,
    ) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_slice_assign(
            tensor.into_float(),
            slices,
            value.into_float(),
        ))
    }

    fn select(tensor: PrimitiveKind, dim: usize, indices: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_select(tensor, dim, indices.into()))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_select(tensor, dim, indices.into()))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn select_assign(
        tensor: PrimitiveKind,
        dim: usize,
        indices: PrimitiveKind,
        values: PrimitiveKind,
        update: IndexingUpdateOp,
    ) -> PrimitiveKind {
        // Select assign is ambiguous for QFloat
        match update {
            IndexingUpdateOp::Add => PrimitiveKind::Float(Dispatch::float_select_add(
                tensor.into_float(),
                dim,
                indices.into(),
                values.into_float(),
            )),
            other => unimplemented!("Unsupported update op {other:?}"),
        }
    }

    fn mask_where(
        tensor: PrimitiveKind,
        mask: PrimitiveKind,
        source: PrimitiveKind,
    ) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_mask_where(
            tensor.into_float(),
            mask.into(),
            source.into_float(),
        ))
    }

    fn mask_fill(tensor: PrimitiveKind, mask: PrimitiveKind, value: Scalar) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_mask_fill(
            tensor.into_float(),
            mask.into(),
            value,
        ))
    }

    fn gather(dim: usize, tensor: PrimitiveKind, indices: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_gather(dim, tensor, indices.into()))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_gather(dim, tensor, indices.into()))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn scatter(
        dim: usize,
        tensor: PrimitiveKind,
        indices: PrimitiveKind,
        values: PrimitiveKind,
        update: IndexingUpdateOp,
    ) -> PrimitiveKind {
        match update {
            IndexingUpdateOp::Add => PrimitiveKind::Float(Dispatch::float_scatter_add(
                dim,
                tensor.into_float(),
                indices.into(),
                values.into_float(),
            )),
            other => unimplemented!("Unsupported update op {other:?}"),
        }
    }

    fn scatter_nd(
        data: PrimitiveKind,
        indices: PrimitiveKind,
        values: PrimitiveKind,
        reduction: IndexingUpdateOp,
    ) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_scatter_nd(
            data.into_float(),
            indices.into(),
            values.into_float(),
            reduction,
        ))
    }

    fn gather_nd(data: PrimitiveKind, indices: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_gather_nd(data.into_float(), indices.into()))
    }

    fn device(tensor: &PrimitiveKind) -> Device {
        match tensor {
            PrimitiveKind::Float(tensor) => Dispatch::float_device(tensor).into(),
            PrimitiveKind::QFloat(tensor) => Dispatch::q_device(tensor).into(),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn to_device(tensor: PrimitiveKind, device: &Device) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_to_device(tensor, &device.dispatch))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_to_device(tensor, &device.dispatch))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    async fn into_data_async(tensor: PrimitiveKind) -> Result<TensorData, ExecutionError> {
        match tensor {
            PrimitiveKind::Float(tensor) => Dispatch::float_into_data(tensor).await,
            PrimitiveKind::QFloat(tensor) => Dispatch::q_into_data(tensor).await,
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn from_data(data: TensorData, device: &Device, dtype: DType) -> PrimitiveKind {
        if matches!(data.dtype, DType::QFloat(_)) {
            // When the source is QFloat, there is no conversion path possible.
            PrimitiveKind::QFloat(Dispatch::q_from_data(data, &device.dispatch))
        } else if dtype.is_float() {
            PrimitiveKind::Float(Dispatch::float_from_data(
                data.convert_dtype(dtype),
                &device.dispatch,
            ))
        } else {
            panic!("Expected float dtype, got {dtype:?}")
        }
    }

    fn repeat_dim(tensor: PrimitiveKind, dim: usize, times: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_repeat_dim(tensor, dim, times))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_repeat_dim(tensor, dim, times))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cat(vectors: Vec<PrimitiveKind>, dim: usize) -> PrimitiveKind {
        match vectors.first().unwrap() {
            PrimitiveKind::Float(_) => PrimitiveKind::Float(Dispatch::float_cat(
                PrimitiveKind::into_dispatch_vec(vectors),
                dim,
            )),
            PrimitiveKind::QFloat(_) => PrimitiveKind::QFloat(Dispatch::q_cat(
                PrimitiveKind::into_dispatch_vec(vectors),
                dim,
            )),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn equal(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_equal(lhs, rhs.into_float(), out_dtype))
    }

    fn not_equal(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_not_equal(lhs, rhs.into_float(), out_dtype))
    }

    fn equal_elem(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_equal_elem(lhs, rhs, out_dtype))
    }

    fn not_equal_elem(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_not_equal_elem(lhs, rhs, out_dtype))
    }

    fn any(tensor: PrimitiveKind) -> PrimitiveKind {
        let tensor = tensor.into_float();
        let out_dtype = Dispatch::float_device(&tensor).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_any(tensor, out_dtype))
    }

    fn any_dim(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        let tensor = tensor.into_float();
        let out_dtype = Dispatch::float_device(&tensor).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_any_dim(tensor, dim, out_dtype))
    }

    fn all(tensor: PrimitiveKind) -> PrimitiveKind {
        let tensor = tensor.into_float();
        let out_dtype = Dispatch::float_device(&tensor).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_all(tensor, out_dtype))
    }

    fn all_dim(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        let tensor = tensor.into_float();
        let out_dtype = Dispatch::float_device(&tensor).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_all_dim(tensor, dim, out_dtype))
    }

    fn permute(tensor: PrimitiveKind, axes: &[usize]) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_permute(tensor, axes))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_permute(tensor, axes))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn expand(tensor: PrimitiveKind, shape: Shape) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_expand(tensor.into_float(), shape))
    }

    fn flip(tensor: PrimitiveKind, axes: &[usize]) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_flip(tensor, axes))
            }
            PrimitiveKind::QFloat(tensor) => PrimitiveKind::QFloat(Dispatch::q_flip(tensor, axes)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn unfold(tensor: PrimitiveKind, dim: usize, size: usize, step: usize) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_unfold(tensor.into_float(), dim, size, step))
    }
}

impl Numeric for Float {
    fn add(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        q_bin_ops!(lhs, rhs, float_add, q_add)
    }

    fn add_scalar(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        match lhs {
            PrimitiveKind::Float(lhs) => PrimitiveKind::Float(Dispatch::float_add_scalar(lhs, rhs)),
            PrimitiveKind::QFloat(lhs) => match Dispatch::q_add_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sub(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        q_bin_ops!(lhs, rhs, float_sub, q_sub)
    }

    fn sub_scalar(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        match lhs {
            PrimitiveKind::Float(lhs) => PrimitiveKind::Float(Dispatch::float_sub_scalar(lhs, rhs)),
            PrimitiveKind::QFloat(lhs) => match Dispatch::q_sub_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn div(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        q_bin_ops!(lhs, rhs, float_div, q_div)
    }

    fn div_scalar(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        match lhs {
            PrimitiveKind::Float(lhs) => PrimitiveKind::Float(Dispatch::float_div_scalar(lhs, rhs)),
            PrimitiveKind::QFloat(lhs) => match Dispatch::q_div_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }
    fn remainder(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_remainder(
            lhs.into_float(),
            rhs.into_float(),
        ))
    }

    fn remainder_scalar(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_remainder_scalar(lhs.into_float(), rhs))
    }

    fn mul(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        q_bin_ops!(lhs, rhs, float_mul, q_mul)
    }

    fn mul_scalar(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        match lhs {
            PrimitiveKind::Float(lhs) => PrimitiveKind::Float(Dispatch::float_mul_scalar(lhs, rhs)),
            PrimitiveKind::QFloat(lhs) => match Dispatch::q_mul_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }
    fn neg(tensor: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => PrimitiveKind::Float(Dispatch::float_neg(tensor)),
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_neg(tensor) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sum(tensor: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => PrimitiveKind::Float(Dispatch::float_sum(tensor)),
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_sum(tensor) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sum_dim(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_sum_dim(tensor, dim))
            }
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_sum_dim(tensor, dim) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn prod(tensor: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => PrimitiveKind::Float(Dispatch::float_prod(tensor)),
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_prod(tensor) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn prod_dim(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_prod_dim(tensor, dim))
            }
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_prod_dim(tensor, dim) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn mean(tensor: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => PrimitiveKind::Float(Dispatch::float_mean(tensor)),
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_mean(tensor) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn mean_dim(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_mean_dim(tensor, dim))
            }
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_mean_dim(tensor, dim) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cumsum(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_cumsum(tensor, dim))
            }
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_cumsum(tensor, dim) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cumprod(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_cumprod(tensor, dim))
            }
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_cumprod(tensor, dim) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn abs(tensor: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => PrimitiveKind::Float(Dispatch::float_abs(tensor)),
            PrimitiveKind::QFloat(tensor) => PrimitiveKind::QFloat(Dispatch::q_abs(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn powi(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        q_bin_ops!(lhs, rhs, float_powf, q_powf)
    }

    fn powi_scalar(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        match lhs {
            PrimitiveKind::Float(lhs) => {
                PrimitiveKind::Float(Dispatch::float_powi_scalar(lhs, rhs))
            }
            PrimitiveKind::QFloat(lhs) => match Dispatch::q_powi_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn random(
        shape: Shape,
        distribution: Distribution,
        device: &Device,
        dtype: DType,
    ) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_random(
            shape,
            distribution,
            &device.dispatch,
            dtype.into(),
        ))
    }

    fn sign(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_sign(tensor.into_float()))
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// # Panics
    ///
    /// If the two tensors don't have a compatible shape.
    fn matmul(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        match (lhs, rhs) {
            (PrimitiveKind::Float(lhs), PrimitiveKind::Float(rhs)) => {
                PrimitiveKind::Float(Dispatch::float_matmul(lhs, rhs))
            }
            (PrimitiveKind::Float(lhs), PrimitiveKind::QFloat(rhs)) => {
                match Dispatch::q_matmul(TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs))
                {
                    TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                    TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
                }
            }
            (PrimitiveKind::QFloat(lhs), PrimitiveKind::Float(rhs)) => {
                match Dispatch::q_matmul(TensorPrimitive::QFloat(lhs), TensorPrimitive::Float(rhs))
                {
                    TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                    TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
                }
            }
            (PrimitiveKind::QFloat(lhs), PrimitiveKind::QFloat(rhs)) => {
                match Dispatch::q_matmul(TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs))
                {
                    TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                    TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
                }
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }
}
impl Ordered for Float {
    fn sort(tensor: PrimitiveKind, dim: usize, descending: bool) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_sort(tensor, dim, descending))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_sort(tensor, dim, descending))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sort_with_indices(
        tensor: PrimitiveKind,
        dim: usize,
        descending: bool,
    ) -> (PrimitiveKind, PrimitiveKind) {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                let (values, indices) =
                    Dispatch::float_sort_with_indices(tensor, dim, descending, out_dtype);
                (PrimitiveKind::Float(values), PrimitiveKind::Int(indices))
            }
            PrimitiveKind::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                let (values, indices) =
                    Dispatch::q_sort_with_indices(tensor, dim, descending, out_dtype);
                (PrimitiveKind::QFloat(values), PrimitiveKind::Int(indices))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn argsort(tensor: PrimitiveKind, dim: usize, descending: bool) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                PrimitiveKind::Int(Dispatch::float_argsort(tensor, dim, descending, out_dtype))
            }
            PrimitiveKind::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                PrimitiveKind::Int(Dispatch::q_argsort(tensor, dim, descending, out_dtype))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cummin(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_cummin(tensor, dim))
            }
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_cummin(tensor, dim) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cummax(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_cummax(tensor, dim))
            }
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_cummax(tensor, dim) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn greater(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_greater(lhs, rhs.into_float(), out_dtype))
    }

    fn greater_elem(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_greater_elem(lhs, rhs, out_dtype))
    }

    fn greater_equal(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_greater_equal(
            lhs,
            rhs.into_float(),
            out_dtype,
        ))
    }

    fn greater_equal_elem(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_greater_equal_elem(lhs, rhs, out_dtype))
    }

    fn lower(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_lower(lhs, rhs.into_float(), out_dtype))
    }

    fn lower_elem(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_lower_elem(lhs, rhs, out_dtype))
    }

    fn lower_equal(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_lower_equal(
            lhs,
            rhs.into_float(),
            out_dtype,
        ))
    }

    fn lower_equal_elem(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        let lhs = lhs.into_float();
        let out_dtype = Dispatch::float_device(&lhs).settings().bool_dtype;
        PrimitiveKind::Bool(Dispatch::float_lower_equal_elem(lhs, rhs, out_dtype))
    }

    fn argmax(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                PrimitiveKind::Int(Dispatch::float_argmax(tensor, dim, out_dtype))
            }
            PrimitiveKind::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                PrimitiveKind::Int(Dispatch::q_argmax(tensor, dim, out_dtype))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn argtopk(tensor: PrimitiveKind, dim: usize, k: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                PrimitiveKind::Int(Dispatch::float_argtopk(tensor, dim, k, out_dtype))
            }
            PrimitiveKind::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                PrimitiveKind::Int(Dispatch::q_argtopk(tensor, dim, k, out_dtype))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn argmin(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                PrimitiveKind::Int(Dispatch::float_argmin(tensor, dim, out_dtype))
            }
            PrimitiveKind::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                PrimitiveKind::Int(Dispatch::q_argmin(tensor, dim, out_dtype))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max(tensor: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => PrimitiveKind::Float(Dispatch::float_max(tensor)),
            PrimitiveKind::QFloat(tensor) => PrimitiveKind::QFloat(Dispatch::q_max(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_dim(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_max_dim(tensor, dim))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_max_dim(tensor, dim))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn topk(tensor: PrimitiveKind, dim: usize, k: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_topk(tensor, dim, k))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_topk(tensor, dim, k))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_dim_with_indices(tensor: PrimitiveKind, dim: usize) -> (PrimitiveKind, PrimitiveKind) {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                let (values, indices) =
                    Dispatch::float_max_dim_with_indices(tensor, dim, out_dtype);
                (PrimitiveKind::Float(values), PrimitiveKind::Int(indices))
            }
            PrimitiveKind::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                let (values, indices) = Dispatch::q_max_dim_with_indices(tensor, dim, out_dtype);
                (PrimitiveKind::QFloat(values), PrimitiveKind::Int(indices))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn min(tensor: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => PrimitiveKind::Float(Dispatch::float_min(tensor)),
            PrimitiveKind::QFloat(tensor) => PrimitiveKind::QFloat(Dispatch::q_min(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn min_dim(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_min_dim(tensor, dim))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_min_dim(tensor, dim))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn min_dim_with_indices(tensor: PrimitiveKind, dim: usize) -> (PrimitiveKind, PrimitiveKind) {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                let out_dtype = Dispatch::float_device(&tensor).settings().int_dtype;
                let (values, indices) =
                    Dispatch::float_min_dim_with_indices(tensor, dim, out_dtype);
                (PrimitiveKind::Float(values), PrimitiveKind::Int(indices))
            }
            PrimitiveKind::QFloat(tensor) => {
                let out_dtype = Dispatch::q_device(&tensor).settings().int_dtype;
                let (values, indices) = Dispatch::q_min_dim_with_indices(tensor, dim, out_dtype);
                (PrimitiveKind::QFloat(values), PrimitiveKind::Int(indices))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn clamp(tensor: PrimitiveKind, min: Scalar, max: Scalar) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_clamp(tensor, min, max))
            }
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_clamp(tensor, min, max) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn clamp_min(tensor: PrimitiveKind, min: Scalar) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_clamp_min(tensor, min))
            }
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_clamp_min(tensor, min) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn clamp_max(tensor: PrimitiveKind, max: Scalar) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_clamp_max(tensor, max))
            }
            PrimitiveKind::QFloat(tensor) => match Dispatch::q_clamp_max(tensor, max) {
                TensorPrimitive::Float(out) => PrimitiveKind::Float(out),
                TensorPrimitive::QFloat(out) => PrimitiveKind::QFloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_abs(tensor: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => PrimitiveKind::Float(Dispatch::float_max_abs(tensor)),
            PrimitiveKind::QFloat(tensor) => PrimitiveKind::QFloat(Dispatch::q_max_abs(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_abs_dim(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => {
                PrimitiveKind::Float(Dispatch::float_max_abs_dim(tensor, dim))
            }
            PrimitiveKind::QFloat(tensor) => {
                PrimitiveKind::QFloat(Dispatch::q_max_abs_dim(tensor, dim))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }
}

impl FloatMathOps for Float {
    fn square(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_powi_scalar(tensor.into_float(), 2.into()))
    }
    fn sqrt(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_sqrt(tensor.into_float()))
    }
    fn cos(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_cos(tensor.into_float()))
    }

    fn sin(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_sin(tensor.into_float()))
    }

    fn tan(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_tan(tensor.into_float()))
    }

    fn cosh(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_cosh(tensor.into_float()))
    }

    fn sinh(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_sinh(tensor.into_float()))
    }

    fn tanh(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_tanh(tensor.into_float()))
    }

    fn acos(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_acos(tensor.into_float()))
    }

    fn acosh(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_acosh(tensor.into_float()))
    }

    fn asin(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_asin(tensor.into_float()))
    }

    fn asinh(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_asinh(tensor.into_float()))
    }

    fn atan(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_atan(tensor.into_float()))
    }

    fn atanh(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_atanh(tensor.into_float()))
    }

    fn atan2(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_atan2(lhs.into_float(), rhs.into_float()))
    }

    fn exp(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_exp(tensor.into_float()))
    }

    fn log(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_log(tensor.into_float()))
    }

    fn log1p(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Float(Dispatch::float_log1p(tensor.into_float()))
    }
}

impl BasicAutodiffOps for Float {
    fn inner(tensor: PrimitiveKind) -> PrimitiveKind {
        match tensor {
            PrimitiveKind::Float(tensor) => PrimitiveKind::Float(Dispatch::inner(tensor)),
            PrimitiveKind::QFloat(tensor) => PrimitiveKind::QFloat(Dispatch::q_inner(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn from_inner(inner: PrimitiveKind) -> PrimitiveKind {
        match inner {
            PrimitiveKind::Float(tensor) => PrimitiveKind::Float(Dispatch::from_inner(tensor)),
            PrimitiveKind::QFloat(tensor) => PrimitiveKind::QFloat(Dispatch::q_from_inner(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}
