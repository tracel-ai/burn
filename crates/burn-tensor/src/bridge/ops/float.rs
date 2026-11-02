use alloc::vec::Vec;
use burn_backend::{
    AutodiffBackend, Distribution, Scalar, TensorData, TensorMetadata, TensorPrimitive,
    distributed::ReduceOperation,
    ops::{
        BoolTensorOps, CollectiveOps, FloatTensorOps, IntTensorOps, QTensorOps,
        TransactionPrimitive,
    },
};
use burn_dispatch::Dispatch;
use burn_std::{DType, ExecutionError, IndexingUpdateOp, PadMode, Shape, Slice};

use crate::{
    Device, Float,
    bridge::{BasicAutodiffOps, BasicOps, FloatMathOps, Numeric, Ordered, TransactionOp},
    ops::{BridgeKind, BridgeTensor, ResolvedOutput},
};

/// A combining operation over two operands, resolved from both: the backend's quantized `$q_op`
/// takes two quantized operands as they are stored, `$op` everything else in the precision the
/// output resolves to.
macro_rules! combine_binary {
    ($lhs:ident, $rhs:ident, $op:ident, $q_op:ident) => {{
        let resolved = ResolvedOutput::combine([&$lhs, &$rhs]);
        match ResolvedOutput::all_quantized([&$lhs, &$rhs]) {
            true => resolved.primitive(Dispatch::$q_op($lhs.into_parts().1, $rhs.into_parts().1)),
            false => resolved.output(Dispatch::$op(
                resolved.operand($lhs),
                resolved.operand($rhs),
            )),
        }
    }};
}

/// A one-operand operation, its output keeping the operand's type: `$quant` runs on a quantized
/// operand as it is stored and returns a [`TensorPrimitive`], `$float` on a float one.
macro_rules! unary {
    ($tensor:ident, |$float_arg:ident| $float:expr, |$quant_arg:ident| $quant:expr) => {{
        let resolved = ResolvedOutput::combine([&$tensor]);
        match $tensor.is_qfloat() {
            true => {
                let $quant_arg = $tensor.into_parts().1;
                resolved.primitive($quant)
            }
            false => {
                let $float_arg = resolved.operand($tensor);
                resolved.output($float)
            }
        }
    }};
    // An operation with no quantized form: a quantized operand is decoded, and the output
    // quantized again when the operand's type is.
    ($tensor:ident, |$float_arg:ident| $float:expr) => {{
        let resolved = ResolvedOutput::combine([&$tensor]);
        let $float_arg = resolved.operand($tensor);
        resolved.output($float)
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
        unary!(
            tensor,
            |tensor| Dispatch::float_reshape(tensor, shape),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_reshape(tensor, shape))
        )
    }

    fn transpose(tensor: BridgeTensor) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_transpose(tensor),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_transpose(tensor))
        )
    }

    fn swap_dims(tensor: BridgeTensor, dim1: usize, dim2: usize) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_swap_dims(tensor, dim1, dim2),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_swap_dims(tensor, dim1, dim2))
        )
    }

    fn slice(tensor: BridgeTensor, slices: &[Slice]) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_slice(tensor, slices),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_slice(tensor, slices))
        )
    }

    fn slice_assign(tensor: BridgeTensor, slices: &[Slice], value: BridgeTensor) -> BridgeTensor {
        let resolved = ResolvedOutput::into(&tensor);
        resolved.output(Dispatch::float_slice_assign(
            resolved.operand(tensor),
            slices,
            resolved.operand(value),
        ))
    }

    fn select(tensor: BridgeTensor, dim: usize, indices: BridgeTensor) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_select(tensor, dim, indices.into()),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_select(tensor, dim, indices.into()))
        )
    }

    fn select_assign(
        tensor: BridgeTensor,
        dim: usize,
        indices: BridgeTensor,
        values: BridgeTensor,
        update: IndexingUpdateOp,
    ) -> BridgeTensor {
        let resolved = ResolvedOutput::into(&tensor);
        resolved.output(Dispatch::float_select_assign(
            resolved.operand(tensor),
            dim,
            indices.into(),
            resolved.operand(values),
            update,
        ))
    }

    fn mask_where(tensor: BridgeTensor, mask: BridgeTensor, source: BridgeTensor) -> BridgeTensor {
        let resolved = ResolvedOutput::combine([&tensor, &source]);
        resolved.output(Dispatch::float_mask_where(
            resolved.operand(tensor),
            mask.into(),
            resolved.operand(source),
        ))
    }

    fn mask_fill(tensor: BridgeTensor, mask: BridgeTensor, value: Scalar) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_mask_fill(
            tensor,
            mask.into(),
            value
        ))
    }

    async fn mask_select(tensor: BridgeTensor, mask: BridgeTensor) -> BridgeTensor {
        let propagation = tensor.propagation();
        let (kind, tensor) = tensor.into_parts();
        let out = match kind {
            BridgeKind::Float => {
                BridgeTensor::float(Dispatch::float_mask_select(tensor, mask.into()).await)
            }
            BridgeKind::QFloat => {
                // Preserve the quantized kind by composing `argwhere` + `q_select` on the
                // quantized primitive, mirroring the default backend op (there is no
                // `q_mask_select`). Using `into_float` here would silently drop quantization.
                let out_dtype = mask.device_settings().int_dtype;
                let n = mask.shape().num_elements();
                let mask = Dispatch::bool_reshape(mask.into(), Shape::new([n]));
                let indices = Dispatch::bool_argwhere(mask, out_dtype).await; // [count, 1]
                let count = indices.shape()[0];
                let indices = Dispatch::int_reshape(indices, Shape::new([count])); // squeeze
                let tensor = Dispatch::q_reshape(tensor, Shape::new([n]));
                BridgeTensor::qfloat(Dispatch::q_select(tensor, 0, indices))
            }
            _ => panic!("Should be Float primitive kind"),
        };
        out.with_propagation(propagation)
    }

    fn gather(dim: usize, tensor: BridgeTensor, indices: BridgeTensor) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_gather(dim, tensor, indices.into()),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_gather(dim, tensor, indices.into()))
        )
    }

    fn scatter(
        dim: usize,
        tensor: BridgeTensor,
        indices: BridgeTensor,
        values: BridgeTensor,
        update: IndexingUpdateOp,
    ) -> BridgeTensor {
        let resolved = ResolvedOutput::into(&tensor);
        resolved.output(Dispatch::float_scatter(
            dim,
            resolved.operand(tensor),
            indices.into(),
            resolved.operand(values),
            update,
        ))
    }

    fn scatter_nd(
        data: BridgeTensor,
        indices: BridgeTensor,
        values: BridgeTensor,
        reduction: IndexingUpdateOp,
    ) -> BridgeTensor {
        let resolved = ResolvedOutput::into(&data);
        resolved.output(Dispatch::float_scatter_nd(
            resolved.operand(data),
            indices.into(),
            resolved.operand(values),
            reduction,
        ))
    }

    fn gather_nd(data: BridgeTensor, indices: BridgeTensor) -> BridgeTensor {
        unary!(data, |data| Dispatch::float_gather_nd(data, indices.into()))
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
        unary!(
            tensor,
            |tensor| Dispatch::float_to_device(tensor, device.as_dispatch()),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_to_device(tensor, device.as_dispatch()))
        )
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
        if matches!(data.dtype(), DType::QFloat(_)) {
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
        unary!(
            tensor,
            |tensor| Dispatch::float_repeat_dim(tensor, dim, times),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_repeat_dim(tensor, dim, times))
        )
    }

    fn cat(vectors: Vec<BridgeTensor>, dim: usize) -> BridgeTensor {
        let resolved = ResolvedOutput::combine(&vectors);
        // Quantized parts concatenate as stored only when they share one scheme.
        let one_scheme = vectors
            .first()
            .is_some_and(|first| vectors.iter().all(|part| part.dtype() == first.dtype()));
        match ResolvedOutput::all_quantized(&vectors) && one_scheme {
            true => resolved.primitive(TensorPrimitive::QFloat(Dispatch::q_cat(
                BridgeTensor::into_dispatch_vec(vectors),
                dim,
            ))),
            false => resolved.output(Dispatch::float_cat(
                vectors
                    .into_iter()
                    .map(|part| resolved.operand(part))
                    .collect(),
                dim,
            )),
        }
    }

    fn equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let resolved = ResolvedOutput::combine([&lhs, &rhs]);
        BridgeTensor::bool(Dispatch::float_equal(
            resolved.operand(lhs),
            resolved.operand(rhs),
            bool_dtype,
        ))
    }

    fn not_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let resolved = ResolvedOutput::combine([&lhs, &rhs]);
        BridgeTensor::bool(Dispatch::float_not_equal(
            resolved.operand(lhs),
            resolved.operand(rhs),
            bool_dtype,
        ))
    }

    fn equal_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_equal_elem(lhs, rhs, bool_dtype))
    }

    fn not_equal_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
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
        unary!(
            tensor,
            |tensor| Dispatch::float_permute(tensor, axes),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_permute(tensor, axes))
        )
    }

    fn expand(tensor: BridgeTensor, shape: Shape) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_expand(tensor, shape))
    }

    fn flip(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_flip(tensor, axes),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_flip(tensor, axes))
        )
    }

    fn unfold(tensor: BridgeTensor, dim: usize, size: usize, step: usize) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_unfold(
            tensor, dim, size, step
        ))
    }

    fn all_gather(tensors: Vec<BridgeTensor>, dim: usize) -> Vec<BridgeTensor> {
        match tensors.first().unwrap().kind() {
            BridgeKind::Float => BridgeTensor::floats(Dispatch::float_all_gather(
                BridgeTensor::into_dispatch_vec(tensors),
                dim,
            )),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn all_to_all(
        tensors: Vec<BridgeTensor>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BridgeTensor> {
        match tensors.first().unwrap().kind() {
            BridgeKind::Float => BridgeTensor::floats(Dispatch::float_all_to_all(
                BridgeTensor::into_dispatch_vec(tensors),
                split_dim,
                concat_dim,
            )),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn broadcast(tensor: BridgeTensor, devices: &[Device]) -> Vec<BridgeTensor> {
        match tensor.kind() {
            BridgeKind::Float => BridgeTensor::floats(Dispatch::float_broadcast(
                tensor.into(),
                &Device::as_dispatch_vec(devices),
            )),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}

impl Numeric for Float {
    fn pad(tensor: BridgeTensor, padding: &[(usize, usize)], mode: PadMode) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_pad(tensor, padding, mode))
    }

    fn add(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        combine_binary!(lhs, rhs, float_add, q_add)
    }

    fn add_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        unary!(lhs, |lhs| Dispatch::float_add_scalar(lhs, rhs), |lhs| {
            Dispatch::q_add_scalar(lhs, rhs)
        })
    }

    fn sub(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        combine_binary!(lhs, rhs, float_sub, q_sub)
    }

    fn sub_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        unary!(lhs, |lhs| Dispatch::float_sub_scalar(lhs, rhs), |lhs| {
            Dispatch::q_sub_scalar(lhs, rhs)
        })
    }

    fn div(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        combine_binary!(lhs, rhs, float_div, q_div)
    }

    fn div_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        unary!(lhs, |lhs| Dispatch::float_div_scalar(lhs, rhs), |lhs| {
            Dispatch::q_div_scalar(lhs, rhs)
        })
    }
    fn remainder(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let resolved = ResolvedOutput::combine([&lhs, &rhs]);
        resolved.output(Dispatch::float_remainder(
            resolved.operand(lhs),
            resolved.operand(rhs),
        ))
    }

    fn remainder_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        unary!(lhs, |lhs| Dispatch::float_remainder_scalar(lhs, rhs))
    }

    fn mul(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        combine_binary!(lhs, rhs, float_mul, q_mul)
    }

    fn mul_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        unary!(lhs, |lhs| Dispatch::float_mul_scalar(lhs, rhs), |lhs| {
            Dispatch::q_mul_scalar(lhs, rhs)
        })
    }
    fn neg(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_neg(tensor), |tensor| {
            Dispatch::q_neg(tensor)
        })
    }

    fn sum(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_sum(tensor), |tensor| {
            Dispatch::q_sum(tensor)
        })
    }

    fn sum_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_sum_dim(tensor, dim),
            |tensor| Dispatch::q_sum_dim(tensor, dim)
        )
    }

    fn sum_dims(tensor: BridgeTensor, dims: &[usize]) -> BridgeTensor {
        match tensor.is_qfloat() {
            // Quantized reductions have no multi-dimension form; one at a time.
            true => dims
                .iter()
                .fold(tensor, |tensor, &dim| Self::sum_dim(tensor, dim)),
            false => unary!(tensor, |tensor| Dispatch::float_sum_dims(tensor, dims)),
        }
    }

    fn prod(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_prod(tensor), |tensor| {
            Dispatch::q_prod(tensor)
        })
    }

    fn prod_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_prod_dim(tensor, dim),
            |tensor| Dispatch::q_prod_dim(tensor, dim)
        )
    }

    fn mean(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_mean(tensor), |tensor| {
            Dispatch::q_mean(tensor)
        })
    }

    fn mean_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_mean_dim(tensor, dim),
            |tensor| Dispatch::q_mean_dim(tensor, dim)
        )
    }

    fn cumsum(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_cumsum(tensor, dim),
            |tensor| Dispatch::q_cumsum(tensor, dim)
        )
    }

    fn cumprod(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_cumprod(tensor, dim),
            |tensor| Dispatch::q_cumprod(tensor, dim)
        )
    }

    fn abs(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_abs(tensor), |tensor| {
            TensorPrimitive::QFloat(Dispatch::q_abs(tensor))
        })
    }

    fn powi(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        combine_binary!(lhs, rhs, float_powf, q_powf)
    }

    fn powi_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        unary!(lhs, |lhs| Dispatch::float_powi_scalar(lhs, rhs), |lhs| {
            Dispatch::q_powi_scalar(lhs, rhs)
        })
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
        unary!(tensor, |tensor| Dispatch::float_sign(tensor))
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// A quantized operand reaches the backend as it is stored, so its kernel reads the packed
    /// values; a float operand is read in the precision the output resolves to, or as it is
    /// beside a quantized one, whose kernel names the precision it multiplies in.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have a compatible shape.
    fn matmul(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let resolved = ResolvedOutput::combine([&lhs, &rhs]);
        let operand = |operand: BridgeTensor| match operand.into_parts() {
            (BridgeKind::Float, operand) => TensorPrimitive::Float(operand),
            (BridgeKind::QFloat, operand) => TensorPrimitive::QFloat(operand),
            _ => panic!("Should be Float primitive kind"),
        };
        match lhs.is_qfloat() || rhs.is_qfloat() {
            true => resolved.primitive(Dispatch::q_matmul(operand(lhs), operand(rhs))),
            false => resolved.output(Dispatch::float_matmul(
                resolved.operand(lhs),
                resolved.operand(rhs),
            )),
        }
    }

    fn all_reduce(tensors: Vec<BridgeTensor>, op: ReduceOperation) -> Vec<BridgeTensor> {
        match tensors.first().unwrap().kind() {
            BridgeKind::Float => BridgeTensor::floats(Dispatch::float_all_reduce(
                BridgeTensor::into_dispatch_vec(tensors),
                op,
            )),
            _ => panic!("Should be Float primitive kind"),
        }
    }

    fn reduce_scatter(
        tensors: Vec<BridgeTensor>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<BridgeTensor> {
        match tensors.first().unwrap().kind() {
            BridgeKind::Float => BridgeTensor::floats(Dispatch::float_reduce_scatter(
                BridgeTensor::into_dispatch_vec(tensors),
                dim,
                op,
            )),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}
impl Ordered for Float {
    fn sort(tensor: BridgeTensor, dim: usize, descending: bool) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_sort(tensor, dim, descending),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_sort(tensor, dim, descending))
        )
    }

    fn sort_with_indices(
        tensor: BridgeTensor,
        dim: usize,
        descending: bool,
    ) -> (BridgeTensor, BridgeTensor) {
        let settings = tensor.device_settings();
        let resolved = ResolvedOutput::combine([&tensor]);
        let (values, indices) = match tensor.into_parts() {
            (BridgeKind::Float, tensor) => {
                let (values, indices) =
                    Dispatch::float_sort_with_indices(tensor, dim, descending, settings.int_dtype);
                (TensorPrimitive::Float(values), indices)
            }
            (BridgeKind::QFloat, tensor) => {
                let (values, indices) =
                    Dispatch::q_sort_with_indices(tensor, dim, descending, settings.int_dtype);
                (TensorPrimitive::QFloat(values), indices)
            }
            _ => panic!("Should be Float primitive kind"),
        };
        (resolved.primitive(values), BridgeTensor::int(indices))
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
        unary!(
            tensor,
            |tensor| Dispatch::float_cummin(tensor, dim),
            |tensor| Dispatch::q_cummin(tensor, dim)
        )
    }

    fn cummax(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_cummax(tensor, dim),
            |tensor| Dispatch::q_cummax(tensor, dim)
        )
    }

    fn greater(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let resolved = ResolvedOutput::combine([&lhs, &rhs]);
        BridgeTensor::bool(Dispatch::float_greater(
            resolved.operand(lhs),
            resolved.operand(rhs),
            bool_dtype,
        ))
    }

    fn greater_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_greater_elem(lhs, rhs, bool_dtype))
    }

    fn greater_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let resolved = ResolvedOutput::combine([&lhs, &rhs]);
        BridgeTensor::bool(Dispatch::float_greater_equal(
            resolved.operand(lhs),
            resolved.operand(rhs),
            bool_dtype,
        ))
    }

    fn greater_equal_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_greater_equal_elem(lhs, rhs, bool_dtype))
    }

    fn lower(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let resolved = ResolvedOutput::combine([&lhs, &rhs]);
        BridgeTensor::bool(Dispatch::float_lower(
            resolved.operand(lhs),
            resolved.operand(rhs),
            bool_dtype,
        ))
    }

    fn lower_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let lhs = lhs.into_float();
        BridgeTensor::bool(Dispatch::float_lower_elem(lhs, rhs, bool_dtype))
    }

    fn lower_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        let resolved = ResolvedOutput::combine([&lhs, &rhs]);
        BridgeTensor::bool(Dispatch::float_lower_equal(
            resolved.operand(lhs),
            resolved.operand(rhs),
            bool_dtype,
        ))
    }

    fn lower_equal_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
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
        unary!(tensor, |tensor| Dispatch::float_max(tensor), |tensor| {
            TensorPrimitive::QFloat(Dispatch::q_max(tensor))
        })
    }

    fn max_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_max_dim(tensor, dim),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_max_dim(tensor, dim))
        )
    }

    fn topk(tensor: BridgeTensor, dim: usize, k: usize) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_topk(tensor, dim, k),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_topk(tensor, dim, k))
        )
    }

    fn topk_with_indices(
        tensor: BridgeTensor,
        dim: usize,
        k: usize,
    ) -> (BridgeTensor, BridgeTensor) {
        let settings = tensor.device_settings();
        let resolved = ResolvedOutput::combine([&tensor]);
        let (values, indices) = match tensor.into_parts() {
            (BridgeKind::Float, tensor) => {
                let (values, indices) =
                    Dispatch::float_topk_with_indices(tensor, dim, k, settings.int_dtype);
                (TensorPrimitive::Float(values), indices)
            }
            (BridgeKind::QFloat, tensor) => {
                let (values, indices) =
                    Dispatch::q_topk_with_indices(tensor, dim, k, settings.int_dtype);
                (TensorPrimitive::QFloat(values), indices)
            }
            _ => panic!("Should be Float primitive kind"),
        };
        (resolved.primitive(values), BridgeTensor::int(indices))
    }

    fn max_dim_with_indices(tensor: BridgeTensor, dim: usize) -> (BridgeTensor, BridgeTensor) {
        let settings = tensor.device_settings();
        let resolved = ResolvedOutput::combine([&tensor]);
        let (values, indices) = match tensor.into_parts() {
            (BridgeKind::Float, tensor) => {
                let (values, indices) =
                    Dispatch::float_max_dim_with_indices(tensor, dim, settings.int_dtype);
                (TensorPrimitive::Float(values), indices)
            }
            (BridgeKind::QFloat, tensor) => {
                let (values, indices) =
                    Dispatch::q_max_dim_with_indices(tensor, dim, settings.int_dtype);
                (TensorPrimitive::QFloat(values), indices)
            }
            _ => panic!("Should be Float primitive kind"),
        };
        (resolved.primitive(values), BridgeTensor::int(indices))
    }

    fn min(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_min(tensor), |tensor| {
            TensorPrimitive::QFloat(Dispatch::q_min(tensor))
        })
    }

    fn min_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_min_dim(tensor, dim),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_min_dim(tensor, dim))
        )
    }

    fn min_dim_with_indices(tensor: BridgeTensor, dim: usize) -> (BridgeTensor, BridgeTensor) {
        let settings = tensor.device_settings();
        let resolved = ResolvedOutput::combine([&tensor]);
        let (values, indices) = match tensor.into_parts() {
            (BridgeKind::Float, tensor) => {
                let (values, indices) =
                    Dispatch::float_min_dim_with_indices(tensor, dim, settings.int_dtype);
                (TensorPrimitive::Float(values), indices)
            }
            (BridgeKind::QFloat, tensor) => {
                let (values, indices) =
                    Dispatch::q_min_dim_with_indices(tensor, dim, settings.int_dtype);
                (TensorPrimitive::QFloat(values), indices)
            }
            _ => panic!("Should be Float primitive kind"),
        };
        (resolved.primitive(values), BridgeTensor::int(indices))
    }

    fn clamp(tensor: BridgeTensor, min: Scalar, max: Scalar) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_clamp(tensor, min, max),
            |tensor| Dispatch::q_clamp(tensor, min, max)
        )
    }

    fn clamp_min(tensor: BridgeTensor, min: Scalar) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_clamp_min(tensor, min),
            |tensor| Dispatch::q_clamp_min(tensor, min)
        )
    }

    fn clamp_max(tensor: BridgeTensor, max: Scalar) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_clamp_max(tensor, max),
            |tensor| Dispatch::q_clamp_max(tensor, max)
        )
    }

    fn max_abs(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_max_abs(tensor), |tensor| {
            TensorPrimitive::QFloat(Dispatch::q_max_abs(tensor))
        })
    }

    fn max_abs_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        unary!(
            tensor,
            |tensor| Dispatch::float_max_abs_dim(tensor, dim),
            |tensor| TensorPrimitive::QFloat(Dispatch::q_max_abs_dim(tensor, dim))
        )
    }
}

impl FloatMathOps for Float {
    fn square(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_powi_scalar(
            tensor,
            2.into()
        ))
    }
    fn sqrt(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_sqrt(tensor))
    }
    fn cos(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_cos(tensor))
    }

    fn sin(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_sin(tensor))
    }

    fn tan(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_tan(tensor))
    }

    fn cosh(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_cosh(tensor))
    }

    fn sinh(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_sinh(tensor))
    }

    fn tanh(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_tanh(tensor))
    }

    fn acos(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_acos(tensor))
    }

    fn acosh(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_acosh(tensor))
    }

    fn asin(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_asin(tensor))
    }

    fn asinh(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_asinh(tensor))
    }

    fn atan(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_atan(tensor))
    }

    fn atanh(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_atanh(tensor))
    }

    fn atan2(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let resolved = ResolvedOutput::combine([&lhs, &rhs]);
        resolved.output(Dispatch::float_atan2(
            resolved.operand(lhs),
            resolved.operand(rhs),
        ))
    }

    fn exp(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_exp(tensor))
    }

    fn log(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_log(tensor))
    }

    fn log1p(tensor: BridgeTensor) -> BridgeTensor {
        unary!(tensor, |tensor| Dispatch::float_log1p(tensor))
    }
}

impl BasicAutodiffOps for Float {
    fn inner(tensor: BridgeTensor) -> BridgeTensor {
        let propagation = tensor.propagation();
        let (kind, tensor) = tensor.into_parts();
        let inner = match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::inner(tensor)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_inner(tensor)),
            _ => panic!("Should be Float primitive kind"),
        };
        inner.with_propagation(propagation)
    }

    fn from_inner(inner: BridgeTensor) -> BridgeTensor {
        let propagation = inner.propagation();
        let (kind, tensor) = inner.into_parts();
        let tensor = match kind {
            BridgeKind::Float => BridgeTensor::float(Dispatch::from_inner(tensor)),
            BridgeKind::QFloat => BridgeTensor::qfloat(Dispatch::q_from_inner(tensor)),
            _ => panic!("Should be Float primitive kind"),
        };
        tensor.with_propagation(propagation)
    }
}
