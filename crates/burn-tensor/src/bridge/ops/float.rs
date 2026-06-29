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
    ops::{BridgeKind, BridgeTensor},
};

fn from_q_primitive(prim: TensorPrimitive<Dispatch>) -> BridgeTensor {
    match prim {
        TensorPrimitive::Float(out) => BridgeTensor::float(out),
        TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
    }
}

macro_rules! q_bin_ops {
    ($lhs:ident, $rhs:ident, $op:ident, $q_op:ident) => {{
        let (lkind, lhs) = $lhs.into_parts();
        let (rkind, rhs) = $rhs.into_parts();
        match (lkind, rkind) {
            (BridgeKind::Float, BridgeKind::Float) => BridgeTensor::float(Dispatch::$op(lhs, rhs)),
            (BridgeKind::QFloat, BridgeKind::QFloat) => from_q_primitive(Dispatch::$q_op(lhs, rhs)),
            (BridgeKind::QFloat, BridgeKind::Float) => {
                let dtype = rhs.dtype();
                BridgeTensor::float(Dispatch::$op(Dispatch::dequantize(lhs, dtype.into()), rhs))
            }
            (BridgeKind::Float, BridgeKind::QFloat) => {
                let dtype = lhs.dtype();
                BridgeTensor::float(Dispatch::$op(lhs, Dispatch::dequantize(rhs, dtype.into())))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }};
}
impl TransactionOp for Float {
    fn register_transaction(tr: &mut TransactionPrimitive<Dispatch>, tensor: BridgeTensor) {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => tr.register_float(TensorPrimitive::Float(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}

impl BasicOps for Float {
    fn empty(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_empty(
            shape,
            device.as_dispatch(),
            dtype.into(),
        ))
    }

    fn zeros(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_zeros(
            shape,
            device.as_dispatch(),
            dtype.into(),
        ))
    }
    fn ones(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_ones(
            shape,
            device.as_dispatch(),
            dtype.into(),
        ))
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_full(
            shape,
            fill_value,
            device.as_dispatch(),
            dtype.into(),
        ))
    }

    fn reshape(tensor: BridgeTensor, shape: Shape) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_reshape(tensor, shape)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_reshape(tensor, shape)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn transpose(tensor: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_transpose(tensor)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_transpose(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn swap_dims(tensor: BridgeTensor, dim1: usize, dim2: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_swap_dims(tensor, dim1, dim2)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_swap_dims(tensor, dim1, dim2)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn slice(tensor: BridgeTensor, slices: &[Slice]) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_slice(tensor, slices)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_slice(tensor, slices)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn slice_assign(tensor: BridgeTensor, slices: &[Slice], value: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_slice_assign(
            tensor.into_float(),
            slices,
            value.into_float(),
        ))
    }

    fn select(tensor: BridgeTensor, dim: usize, indices: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => {
                BridgeTensor::float(Dispatch::float_select(tensor, dim, indices.into()))
            }
            BridgeKind::QFloat => {
                BridgeTensor::qfloat(Dispatch::q_select(tensor, dim, indices.into()))
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
            IndexingUpdateOp::Add => BridgeTensor::float(Dispatch::float_select_add(
                tensor.into_float(),
                dim,
                indices.into(),
                values.into_float(),
            )),
            other => unimplemented!("Unsupported update op {other:?}"),
        }
    }

    fn mask_where(tensor: BridgeTensor, mask: BridgeTensor, source: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_mask_where(
            tensor.into_float(),
            mask.into(),
            source.into_float(),
        ))
    }

    fn mask_fill(tensor: BridgeTensor, mask: BridgeTensor, value: Scalar) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_mask_fill(
            tensor.into_float(),
            mask.into(),
            value,
        ))
    }

    fn gather(dim: usize, tensor: BridgeTensor, indices: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => {
                BridgeTensor::float(Dispatch::float_gather(dim, tensor, indices.into()))
            }
            BridgeKind::QFloat => {
                BridgeTensor::qfloat(Dispatch::q_gather(dim, tensor, indices.into()))
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
            IndexingUpdateOp::Add => BridgeTensor::float(Dispatch::float_scatter_add(
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
        BridgeTensor::float(Dispatch::float_scatter_nd(
            data.into_float(),
            indices.into(),
            values.into_float(),
            reduction,
        ))
    }

    fn gather_nd(data: BridgeTensor, indices: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_gather_nd(data.into_float(), indices.into()))
    }

    fn device(tensor: &BridgeTensor) -> Device {
        let (kind, tensor) = tensor.as_parts();
        match kind {
            BridgeKind::Float => Device::new(tensor.device()),
            BridgeKind::QFloat => Device::new(tensor.device()),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn to_device(tensor: BridgeTensor, device: &Device) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => {
                BridgeTensor::float(Dispatch::float_to_device(tensor, device.as_dispatch()))
            }
            BridgeKind::QFloat => {
                BridgeTensor::qfloat(Dispatch::q_to_device(tensor, device.as_dispatch()))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    async fn into_data_async(tensor: BridgeTensor) -> Result<TensorData, ExecutionError> {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => Dispatch::float_into_data(tensor).await,
            BridgeKind::QFloat => Dispatch::q_into_data(tensor).await,
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn from_data(data: TensorData, device: &Device, dtype: DType) -> BridgeTensor {
        if matches!(data.dtype, DType::QFloat(_)) {
            // When the source is QFloat, there is no conversion path possible.
            BridgeTensor::qfloat(Dispatch::q_from_data(data, device.as_dispatch()))
        } else if dtype.is_float() {
            BridgeTensor::float(Dispatch::float_from_data(
                data.convert_dtype(dtype),
                device.as_dispatch(),
            ))
        } else {
            panic!("Expected float dtype, got {dtype:?}")
        }
    }

    fn repeat_dim(tensor: BridgeTensor, dim: usize, times: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => {
                BridgeTensor::float(Dispatch::float_repeat_dim(tensor, dim, times))
            }
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_repeat_dim(tensor, dim, times)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cat(vectors: Vec<BridgeTensor>, dim: usize) -> BridgeTensor {
        match vectors.first().unwrap().kind() {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_cat(
                BridgeTensor::into_dispatch_vec(vectors),
                dim,
            )),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_cat(
                BridgeTensor::into_dispatch_vec(vectors),
                dim,
            )),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_equal(lhs, rhs.into_float(), bool_dtype))
    }

    fn not_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_not_equal(lhs, rhs.into_float(), bool_dtype))
    }

    fn equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_equal_elem(lhs, rhs, bool_dtype))
    }

    fn not_equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_not_equal_elem(lhs, rhs, bool_dtype))
    }

    fn any(tensor: BridgeTensor) -> BridgeTensor {
        let bool_dtype = tensor.device_settings().bool_dtype;
        let tensor = tensor.into_float();
        BridgeTensor::bool(Dispatch::float_any(tensor, bool_dtype))
    }

    fn any_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let bool_dtype = tensor.device_settings().bool_dtype;
        let tensor = tensor.into_float();
        BridgeTensor::bool(Dispatch::float_any_dim(tensor, dim, bool_dtype))
    }

    fn all(tensor: BridgeTensor) -> BridgeTensor {
        let bool_dtype = tensor.device_settings().bool_dtype;
        let tensor = tensor.into_float();
        BridgeTensor::bool(Dispatch::float_all(tensor, bool_dtype))
    }

    fn all_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let bool_dtype = tensor.device_settings().bool_dtype;
        let tensor = tensor.into_float();
        BridgeTensor::bool(Dispatch::float_all_dim(tensor, dim, bool_dtype))
    }

    fn permute(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_permute(tensor, axes)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_permute(tensor, axes)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn expand(tensor: BridgeTensor, shape: Shape) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_expand(tensor.into_float(), shape))
    }

    fn flip(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_flip(tensor, axes)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_flip(tensor, axes)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn unfold(tensor: BridgeTensor, dim: usize, size: usize, step: usize) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_unfold(tensor.into_float(), dim, size, step))
    }
}

impl Numeric for Float {
    fn add(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        q_bin_ops!(lhs, rhs, float_add, q_add)
    }

    fn add_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let (kind, lhs) = lhs.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_add_scalar(lhs, rhs)),
            BridgeKind::QFloat => match Dispatch::q_add_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sub(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        q_bin_ops!(lhs, rhs, float_sub, q_sub)
    }

    fn sub_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let (kind, lhs) = lhs.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_sub_scalar(lhs, rhs)),
            BridgeKind::QFloat => match Dispatch::q_sub_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn div(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        q_bin_ops!(lhs, rhs, float_div, q_div)
    }

    fn div_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let (kind, lhs) = lhs.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_div_scalar(lhs, rhs)),
            BridgeKind::QFloat => match Dispatch::q_div_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }
    fn remainder(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_remainder(
            lhs.into_float(),
            rhs.into_float(),
        ))
    }

    fn remainder_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_remainder_scalar(lhs.into_float(), rhs))
    }

    fn mul(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        q_bin_ops!(lhs, rhs, float_mul, q_mul)
    }

    fn mul_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let (kind, lhs) = lhs.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_mul_scalar(lhs, rhs)),
            BridgeKind::QFloat => match Dispatch::q_mul_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }
    fn neg(tensor: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_neg(tensor)),
            BridgeKind::QFloat => match Dispatch::q_neg(tensor) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sum(tensor: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_sum(tensor)),
            BridgeKind::QFloat => match Dispatch::q_sum(tensor) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sum_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_sum_dim(tensor, dim)),
            BridgeKind::QFloat => match Dispatch::q_sum_dim(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn prod(tensor: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_prod(tensor)),
            BridgeKind::QFloat => match Dispatch::q_prod(tensor) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn prod_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_prod_dim(tensor, dim)),
            BridgeKind::QFloat => match Dispatch::q_prod_dim(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn mean(tensor: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_mean(tensor)),
            BridgeKind::QFloat => match Dispatch::q_mean(tensor) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn mean_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_mean_dim(tensor, dim)),
            BridgeKind::QFloat => match Dispatch::q_mean_dim(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cumsum(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_cumsum(tensor, dim)),
            BridgeKind::QFloat => match Dispatch::q_cumsum(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cumprod(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_cumprod(tensor, dim)),
            BridgeKind::QFloat => match Dispatch::q_cumprod(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn powi(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        q_bin_ops!(lhs, rhs, float_powf, q_powf)
    }

    fn powi_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let (kind, lhs) = lhs.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_powi_scalar(lhs, rhs)),
            BridgeKind::QFloat => match Dispatch::q_powi_scalar(lhs, rhs) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
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
        BridgeTensor::float(Dispatch::float_random(
            shape,
            distribution,
            device.as_dispatch(),
            dtype.into(),
        ))
    }

    fn sign(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_sign(tensor.into_float()))
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// # Panics
    ///
    /// If the two tensors don't have a compatible shape.
    fn matmul(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let (lkind, lhs) = lhs.into_parts();
        let (rkind, rhs) = rhs.into_parts();
        match (lkind, rkind) {
            (BridgeKind::Float, BridgeKind::Float) => {
                BridgeTensor::float(Dispatch::float_matmul(lhs, rhs))
            }
            (BridgeKind::Float, BridgeKind::QFloat) => from_q_primitive(Dispatch::q_matmul(
                TensorPrimitive::Float(lhs),
                TensorPrimitive::QFloat(rhs),
            )),
            (BridgeKind::QFloat, BridgeKind::Float) => from_q_primitive(Dispatch::q_matmul(
                TensorPrimitive::QFloat(lhs),
                TensorPrimitive::Float(rhs),
            )),
            (BridgeKind::QFloat, BridgeKind::QFloat) => from_q_primitive(Dispatch::q_matmul(
                TensorPrimitive::QFloat(lhs),
                TensorPrimitive::QFloat(rhs),
            )),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}
impl Ordered for Float {
    fn abs(tensor: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_abs(tensor)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_abs(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }
    fn sort(tensor: BridgeTensor, dim: usize, descending: bool) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_sort(tensor, dim, descending)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_sort(tensor, dim, descending)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn sort_with_indices(
        tensor: BridgeTensor,
        dim: usize,
        descending: bool,
    ) -> (BridgeTensor, BridgeTensor) {
        let settings = tensor.device_settings();
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => {
                let (values, indices) =
                    Dispatch::float_sort_with_indices(tensor, dim, descending, settings.int_dtype);
                (BridgeTensor::float(values), BridgeTensor::int(indices))
            }
            BridgeKind::QFloat => {
                let (values, indices) =
                    Dispatch::q_sort_with_indices(tensor, dim, descending, settings.int_dtype);
                (BridgeTensor::qfloat(values), BridgeTensor::int(indices))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn argsort(tensor: BridgeTensor, dim: usize, descending: bool) -> BridgeTensor {
        let settings = tensor.device_settings();
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::int(Dispatch::float_argsort(
                tensor,
                dim,
                descending,
                settings.int_dtype,
            )),
            BridgeKind::QFloat => BridgeTensor::int(Dispatch::q_argsort(
                tensor,
                dim,
                descending,
                settings.int_dtype,
            )),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cummin(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_cummin(tensor, dim)),
            BridgeKind::QFloat => match Dispatch::q_cummin(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn cummax(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_cummax(tensor, dim)),
            BridgeKind::QFloat => match Dispatch::q_cummax(tensor, dim) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn greater(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_greater(lhs, rhs.into_float(), bool_dtype))
    }

    fn greater_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_greater_elem(lhs, rhs, bool_dtype))
    }

    fn greater_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_greater_equal(
            lhs,
            rhs.into_float(),
            bool_dtype,
        ))
    }

    fn greater_equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_greater_equal_elem(lhs, rhs, bool_dtype))
    }

    fn lower(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_lower(lhs, rhs.into_float(), bool_dtype))
    }

    fn lower_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_lower_elem(lhs, rhs, bool_dtype))
    }

    fn lower_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_lower_equal(
            lhs,
            rhs.into_float(),
            bool_dtype,
        ))
    }

    fn lower_equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_lower_equal_elem(lhs, rhs, bool_dtype))
    }

    fn argmax(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let settings = tensor.device_settings();
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => {
                BridgeTensor::int(Dispatch::float_argmax(tensor, dim, settings.int_dtype))
            }
            BridgeKind::QFloat => {
                BridgeTensor::int(Dispatch::q_argmax(tensor, dim, settings.int_dtype))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn argtopk(tensor: BridgeTensor, dim: usize, k: usize) -> BridgeTensor {
        let settings = tensor.device_settings();
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => {
                BridgeTensor::int(Dispatch::float_argtopk(tensor, dim, k, settings.int_dtype))
            }
            BridgeKind::QFloat => {
                BridgeTensor::int(Dispatch::q_argtopk(tensor, dim, k, settings.int_dtype))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn argmin(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let settings = tensor.device_settings();
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => {
                BridgeTensor::int(Dispatch::float_argmin(tensor, dim, settings.int_dtype))
            }
            BridgeKind::QFloat => {
                BridgeTensor::int(Dispatch::q_argmin(tensor, dim, settings.int_dtype))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max(tensor: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_max(tensor)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_max(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_max_dim(tensor, dim)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_max_dim(tensor, dim)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn topk(tensor: BridgeTensor, dim: usize, k: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_topk(tensor, dim, k)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_topk(tensor, dim, k)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_dim_with_indices(tensor: BridgeTensor, dim: usize) -> (BridgeTensor, BridgeTensor) {
        let settings = tensor.device_settings();
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => {
                let (values, indices) =
                    Dispatch::float_max_dim_with_indices(tensor, dim, settings.int_dtype);
                (BridgeTensor::float(values), BridgeTensor::int(indices))
            }
            BridgeKind::QFloat => {
                let (values, indices) =
                    Dispatch::q_max_dim_with_indices(tensor, dim, settings.int_dtype);
                (BridgeTensor::qfloat(values), BridgeTensor::int(indices))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn min(tensor: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_min(tensor)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_min(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn min_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_min_dim(tensor, dim)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_min_dim(tensor, dim)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn min_dim_with_indices(tensor: BridgeTensor, dim: usize) -> (BridgeTensor, BridgeTensor) {
        let settings = tensor.device_settings();
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => {
                let (values, indices) =
                    Dispatch::float_min_dim_with_indices(tensor, dim, settings.int_dtype);
                (BridgeTensor::float(values), BridgeTensor::int(indices))
            }
            BridgeKind::QFloat => {
                let (values, indices) =
                    Dispatch::q_min_dim_with_indices(tensor, dim, settings.int_dtype);
                (BridgeTensor::qfloat(values), BridgeTensor::int(indices))
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn clamp(tensor: BridgeTensor, min: Scalar, max: Scalar) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_clamp(tensor, min, max)),
            BridgeKind::QFloat => match Dispatch::q_clamp(tensor, min, max) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn clamp_min(tensor: BridgeTensor, min: Scalar) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_clamp_min(tensor, min)),
            BridgeKind::QFloat => match Dispatch::q_clamp_min(tensor, min) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn clamp_max(tensor: BridgeTensor, max: Scalar) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_clamp_max(tensor, max)),
            BridgeKind::QFloat => match Dispatch::q_clamp_max(tensor, max) {
                TensorPrimitive::Float(out) => BridgeTensor::float(out),
                TensorPrimitive::QFloat(out) => BridgeTensor::qfloat(out),
            },
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_abs(tensor: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_max_abs(tensor)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_max_abs(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn max_abs_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::float_max_abs_dim(tensor, dim)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_max_abs_dim(tensor, dim)),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}

impl FloatMathOps for Float {
    fn square(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_powi_scalar(tensor.into_float(), 2.into()))
    }
    fn sqrt(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_sqrt(tensor.into_float()))
    }
    fn cos(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_cos(tensor.into_float()))
    }

    fn sin(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_sin(tensor.into_float()))
    }

    fn tan(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_tan(tensor.into_float()))
    }

    fn cosh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_cosh(tensor.into_float()))
    }

    fn sinh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_sinh(tensor.into_float()))
    }

    fn tanh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_tanh(tensor.into_float()))
    }

    fn acos(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_acos(tensor.into_float()))
    }

    fn acosh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_acosh(tensor.into_float()))
    }

    fn asin(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_asin(tensor.into_float()))
    }

    fn asinh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_asinh(tensor.into_float()))
    }

    fn atan(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_atan(tensor.into_float()))
    }

    fn atanh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_atanh(tensor.into_float()))
    }
    fn atan2(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_atan2(lhs.into_float(), rhs.into_float()))
    }
    fn exp(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_exp(tensor.into_float()))
    }

    fn log(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_log(tensor.into_float()))
    }

    fn log1p(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::float(Dispatch::float_log1p(tensor.into_float()))
    }
}

impl BasicAutodiffOps for Float {
    fn inner(tensor: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = tensor.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::inner(tensor)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_inner(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn from_inner(inner: BridgeTensor) -> BridgeTensor {
        let (kind, tensor) = inner.into_parts();
        match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::from_inner(tensor)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_from_inner(tensor)),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}
