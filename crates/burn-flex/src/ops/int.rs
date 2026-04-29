//! Int tensor operations for the Flex backend.

use alloc::vec::Vec;
use burn_backend::{
    DType, Distribution, ExecutionError, FloatDType, Scalar, TensorData, TensorMetadata,
    ops::IntTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_std::{Bytes, IntDType, Shape, Slice, bf16, f16};
use num_traits::ToPrimitive;

use crate::Layout;
use crate::ops::binary::{binary_op_typed, int_binary_op, int_scalar_op, scalar_op_typed};
use crate::{Flex, FlexTensor, ops::matmul};

/// Convert a Scalar to (i64, u64) pair for the given dtype.
/// Only the matching type's conversion is validated; the other gets a dummy 0.
fn scalar_to_int_pair(dtype: DType, rhs: &Scalar) -> (i64, u64) {
    if dtype == DType::U64 {
        (0, rhs.to_u64().unwrap())
    } else {
        (rhs.to_i64().unwrap(), 0)
    }
}

impl IntTensorOps<Flex> for Flex {
    fn int_from_data(data: TensorData, _device: &Device<Flex>) -> IntTensor<Flex> {
        FlexTensor::from_data(data)
    }

    async fn int_into_data(tensor: IntTensor<Flex>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn int_device(_tensor: &IntTensor<Flex>) -> Device<Flex> {
        Default::default()
    }

    fn int_to_device(tensor: IntTensor<Flex>, _device: &Device<Flex>) -> IntTensor<Flex> {
        tensor
    }

    fn int_cat(tensors: Vec<IntTensor<Flex>>, dim: usize) -> IntTensor<Flex> {
        crate::ops::cat::cat(tensors, dim)
    }

    fn int_reshape(tensor: IntTensor<Flex>, shape: Shape) -> IntTensor<Flex> {
        tensor.reshape(shape)
    }

    fn int_slice(tensor: IntTensor<Flex>, slices: &[Slice]) -> IntTensor<Flex> {
        crate::ops::slice::slice(tensor, slices)
    }

    fn int_empty(shape: Shape, _device: &Device<Flex>, dtype: IntDType) -> IntTensor<Flex> {
        FlexTensor::empty(shape, dtype.into())
    }

    fn int_mask_where(
        tensor: IntTensor<Flex>,
        mask: BoolTensor<Flex>,
        value: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        debug_assert_eq!(
            tensor.dtype(),
            value.dtype(),
            "int_mask_where: dtype mismatch"
        );
        match tensor.dtype() {
            DType::I64 => crate::ops::mask::mask_where::<i64>(tensor, mask, value),
            DType::I32 => crate::ops::mask::mask_where::<i32>(tensor, mask, value),
            DType::I16 => crate::ops::mask::mask_where::<i16>(tensor, mask, value),
            DType::I8 => crate::ops::mask::mask_where::<i8>(tensor, mask, value),
            DType::U64 => crate::ops::mask::mask_where::<u64>(tensor, mask, value),
            DType::U32 => crate::ops::mask::mask_where::<u32>(tensor, mask, value),
            DType::U16 => crate::ops::mask::mask_where::<u16>(tensor, mask, value),
            DType::U8 => crate::ops::mask::mask_where::<u8>(tensor, mask, value),
            dt => panic!("int_mask_where: unsupported dtype {:?}", dt),
        }
    }

    fn int_mask_fill(
        tensor: IntTensor<Flex>,
        mask: BoolTensor<Flex>,
        value: Scalar,
    ) -> IntTensor<Flex> {
        match tensor.dtype() {
            DType::I64 => crate::ops::mask::mask_fill(tensor, mask, value.to_i64().unwrap()),
            DType::I32 => crate::ops::mask::mask_fill(tensor, mask, value.to_i64().unwrap() as i32),
            DType::I16 => crate::ops::mask::mask_fill(tensor, mask, value.to_i64().unwrap() as i16),
            DType::I8 => crate::ops::mask::mask_fill(tensor, mask, value.to_i64().unwrap() as i8),
            DType::U64 => crate::ops::mask::mask_fill(tensor, mask, value.to_u64().unwrap()),
            DType::U32 => crate::ops::mask::mask_fill(tensor, mask, value.to_u64().unwrap() as u32),
            DType::U16 => crate::ops::mask::mask_fill(tensor, mask, value.to_u64().unwrap() as u16),
            DType::U8 => crate::ops::mask::mask_fill(tensor, mask, value.to_u64().unwrap() as u8),
            dt => panic!("int_mask_fill: unsupported dtype {:?}", dt),
        }
    }

    fn int_slice_assign(
        tensor: IntTensor<Flex>,
        slices: &[Slice],
        value: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        crate::ops::slice::slice_assign(tensor, slices, value)
    }

    /// Gather ints along `dim` at the given indices.
    ///
    /// The `tensor` dispatches on its own int dtype (I8/I16/I32/I64 signed or
    /// U8/U16/U32/U64 unsigned). The `indices` tensor may be any of those
    /// widths too - it's normalised to `isize` by the shared `read_indices`
    /// helper in `ops::gather_scatter` before the kernel runs, so callers are
    /// not required to pre-convert to I64.
    fn int_gather(
        dim: usize,
        tensor: IntTensor<Flex>,
        indices: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        match tensor.dtype() {
            DType::I64 => crate::ops::gather_scatter::gather::<i64>(tensor, dim, indices),
            DType::I32 => crate::ops::gather_scatter::gather::<i32>(tensor, dim, indices),
            DType::I16 => crate::ops::gather_scatter::gather::<i16>(tensor, dim, indices),
            DType::I8 => crate::ops::gather_scatter::gather::<i8>(tensor, dim, indices),
            DType::U64 => crate::ops::gather_scatter::gather::<u64>(tensor, dim, indices),
            DType::U32 => crate::ops::gather_scatter::gather::<u32>(tensor, dim, indices),
            DType::U16 => crate::ops::gather_scatter::gather::<u16>(tensor, dim, indices),
            DType::U8 => crate::ops::gather_scatter::gather::<u8>(tensor, dim, indices),
            dt => panic!("int_gather: unsupported dtype {:?}", dt),
        }
    }

    /// Scatter-add int values at the given indices along `dim`.
    ///
    /// `tensor` and `value` must share the same int dtype; `indices` may be
    /// any supported int width. See [`int_gather`](Self::int_gather) for the
    /// full index-width policy.
    fn int_scatter_add(
        dim: usize,
        tensor: IntTensor<Flex>,
        indices: IntTensor<Flex>,
        value: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        debug_assert_eq!(
            tensor.dtype(),
            value.dtype(),
            "int_scatter_add: dtype mismatch"
        );
        match tensor.dtype() {
            DType::I64 => {
                crate::ops::gather_scatter::scatter_add::<i64>(tensor, dim, indices, value)
            }
            DType::I32 => {
                crate::ops::gather_scatter::scatter_add::<i32>(tensor, dim, indices, value)
            }
            DType::I16 => {
                crate::ops::gather_scatter::scatter_add::<i16>(tensor, dim, indices, value)
            }
            DType::I8 => crate::ops::gather_scatter::scatter_add::<i8>(tensor, dim, indices, value),
            DType::U64 => {
                crate::ops::gather_scatter::scatter_add::<u64>(tensor, dim, indices, value)
            }
            DType::U32 => {
                crate::ops::gather_scatter::scatter_add::<u32>(tensor, dim, indices, value)
            }
            DType::U16 => {
                crate::ops::gather_scatter::scatter_add::<u16>(tensor, dim, indices, value)
            }
            DType::U8 => crate::ops::gather_scatter::scatter_add::<u8>(tensor, dim, indices, value),
            dt => panic!("int_scatter_add: unsupported dtype {:?}", dt),
        }
    }

    fn int_scatter_nd(
        data: IntTensor<Flex>,
        indices: IntTensor<Flex>,
        values: IntTensor<Flex>,
        reduction: burn_backend::tensor::IndexingUpdateOp,
    ) -> IntTensor<Flex> {
        match data.dtype() {
            DType::I64 => {
                crate::ops::gather_scatter::scatter_nd::<i64>(data, indices, values, reduction)
            }
            DType::I32 => {
                crate::ops::gather_scatter::scatter_nd::<i32>(data, indices, values, reduction)
            }
            DType::I16 => {
                crate::ops::gather_scatter::scatter_nd::<i16>(data, indices, values, reduction)
            }
            DType::I8 => {
                crate::ops::gather_scatter::scatter_nd::<i8>(data, indices, values, reduction)
            }
            DType::U64 => {
                crate::ops::gather_scatter::scatter_nd::<u64>(data, indices, values, reduction)
            }
            DType::U32 => {
                crate::ops::gather_scatter::scatter_nd::<u32>(data, indices, values, reduction)
            }
            DType::U16 => {
                crate::ops::gather_scatter::scatter_nd::<u16>(data, indices, values, reduction)
            }
            DType::U8 => {
                crate::ops::gather_scatter::scatter_nd::<u8>(data, indices, values, reduction)
            }
            dt => panic!("int_scatter_nd: unsupported dtype {:?}", dt),
        }
    }

    fn int_gather_nd(data: IntTensor<Flex>, indices: IntTensor<Flex>) -> IntTensor<Flex> {
        match data.dtype() {
            DType::I64 => crate::ops::gather_scatter::gather_nd::<i64>(data, indices),
            DType::I32 => crate::ops::gather_scatter::gather_nd::<i32>(data, indices),
            DType::I16 => crate::ops::gather_scatter::gather_nd::<i16>(data, indices),
            DType::I8 => crate::ops::gather_scatter::gather_nd::<i8>(data, indices),
            DType::U64 => crate::ops::gather_scatter::gather_nd::<u64>(data, indices),
            DType::U32 => crate::ops::gather_scatter::gather_nd::<u32>(data, indices),
            DType::U16 => crate::ops::gather_scatter::gather_nd::<u16>(data, indices),
            DType::U8 => crate::ops::gather_scatter::gather_nd::<u8>(data, indices),
            dt => panic!("int_gather_nd: unsupported dtype {:?}", dt),
        }
    }

    /// Select ints along `dim` by a 1D index tensor.
    ///
    /// The `indices` tensor may be any supported int width. See
    /// [`int_gather`](Self::int_gather) for the full index-width policy.
    fn int_select(
        tensor: IntTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        match tensor.dtype() {
            DType::I64 => crate::ops::gather_scatter::select::<i64>(tensor, dim, indices),
            DType::I32 => crate::ops::gather_scatter::select::<i32>(tensor, dim, indices),
            DType::I16 => crate::ops::gather_scatter::select::<i16>(tensor, dim, indices),
            DType::I8 => crate::ops::gather_scatter::select::<i8>(tensor, dim, indices),
            DType::U64 => crate::ops::gather_scatter::select::<u64>(tensor, dim, indices),
            DType::U32 => crate::ops::gather_scatter::select::<u32>(tensor, dim, indices),
            DType::U16 => crate::ops::gather_scatter::select::<u16>(tensor, dim, indices),
            DType::U8 => crate::ops::gather_scatter::select::<u8>(tensor, dim, indices),
            dt => panic!("int_select: unsupported dtype {:?}", dt),
        }
    }

    /// Select-add int values at a 1D index tensor along `dim`.
    ///
    /// `tensor` and `value` must share the same int dtype; `indices` may be
    /// any supported int width. See [`int_gather`](Self::int_gather) for the
    /// full index-width policy.
    fn int_select_add(
        tensor: IntTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
        value: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        debug_assert_eq!(
            tensor.dtype(),
            value.dtype(),
            "int_select_add: dtype mismatch"
        );
        match tensor.dtype() {
            DType::I64 => {
                crate::ops::gather_scatter::select_add::<i64>(tensor, dim, indices, value)
            }
            DType::I32 => {
                crate::ops::gather_scatter::select_add::<i32>(tensor, dim, indices, value)
            }
            DType::I16 => {
                crate::ops::gather_scatter::select_add::<i16>(tensor, dim, indices, value)
            }
            DType::I8 => crate::ops::gather_scatter::select_add::<i8>(tensor, dim, indices, value),
            DType::U64 => {
                crate::ops::gather_scatter::select_add::<u64>(tensor, dim, indices, value)
            }
            DType::U32 => {
                crate::ops::gather_scatter::select_add::<u32>(tensor, dim, indices, value)
            }
            DType::U16 => {
                crate::ops::gather_scatter::select_add::<u16>(tensor, dim, indices, value)
            }
            DType::U8 => crate::ops::gather_scatter::select_add::<u8>(tensor, dim, indices, value),
            dt => panic!("int_select_add: unsupported dtype {:?}", dt),
        }
    }

    fn int_equal(
        lhs: IntTensor<Flex>,
        rhs: IntTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::int_equal(lhs, rhs, out_dtype)
    }

    fn int_equal_elem(
        lhs: IntTensor<Flex>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
        crate::ops::comparison::int_equal_elem(lhs, i, u, out_dtype)
    }

    fn int_greater(
        lhs: IntTensor<Flex>,
        rhs: IntTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::int_greater(lhs, rhs, out_dtype)
    }

    fn int_greater_elem(
        lhs: IntTensor<Flex>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
        crate::ops::comparison::int_greater_elem(lhs, i, u, out_dtype)
    }

    fn int_greater_equal(
        lhs: IntTensor<Flex>,
        rhs: IntTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::int_greater_equal(lhs, rhs, out_dtype)
    }

    fn int_greater_equal_elem(
        lhs: IntTensor<Flex>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
        crate::ops::comparison::int_greater_equal_elem(lhs, i, u, out_dtype)
    }

    fn int_lower(
        lhs: IntTensor<Flex>,
        rhs: IntTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::int_lower(lhs, rhs, out_dtype)
    }

    fn int_lower_elem(
        lhs: IntTensor<Flex>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
        crate::ops::comparison::int_lower_elem(lhs, i, u, out_dtype)
    }

    fn int_lower_equal(
        lhs: IntTensor<Flex>,
        rhs: IntTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::int_lower_equal(lhs, rhs, out_dtype)
    }

    fn int_lower_equal_elem(
        lhs: IntTensor<Flex>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
        crate::ops::comparison::int_lower_equal_elem(lhs, i, u, out_dtype)
    }

    fn int_add(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a + b)
    }

    fn int_add_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        if lhs.dtype() == DType::U64 {
            return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| {
                a.wrapping_add(b)
            });
        }
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a + b)
    }

    fn int_sub(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a - b)
    }

    fn int_sub_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        if lhs.dtype() == DType::U64 {
            return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| {
                a.wrapping_sub(b)
            });
        }
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a - b)
    }

    fn int_mul(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a * b)
    }

    fn int_mul_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        if lhs.dtype() == DType::U64 {
            return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| {
                a.wrapping_mul(b)
            });
        }
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a * b)
    }

    fn int_div(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        // U64 values > i64::MAX produce wrong results through i64 cast
        if lhs.dtype() == DType::U64 {
            let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);
            return binary_op_typed(lhs, &rhs, |a: u64, b: u64| a / b);
        }
        int_binary_op(lhs, rhs, |a, b| a / b)
    }

    fn int_div_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        if lhs.dtype() == DType::U64 {
            return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a / b);
        }
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a / b)
    }

    fn int_remainder(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        // U64 values > i64::MAX produce wrong results through i64 cast
        if lhs.dtype() == DType::U64 {
            let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);
            return binary_op_typed(lhs, &rhs, |a: u64, b: u64| a % b);
        }
        // Python/PyTorch-style remainder: result has same sign as divisor
        int_binary_op(lhs, rhs, |a, b| ((a % b) + b) % b)
    }

    fn int_remainder_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        if lhs.dtype() == DType::U64 {
            return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a % b);
        }
        // Python/PyTorch-style remainder: result has same sign as divisor
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| ((a % b) + b) % b)
    }

    // Precision limits: i64/u64 > 2^24 for f32/f16/bf16, > 2^53 for f64.
    fn int_into_float(
        tensor: IntTensor<Flex>,
        out_dtype: burn_std::FloatDType,
    ) -> FloatTensor<Flex> {
        let tensor = tensor.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let src = tensor.dtype();
        let out_dt = DType::from(out_dtype);

        // Read source ints, applying conversion per-element.
        // Each arm binds `$x` to the native int value; `$conv` must work for all int types.
        macro_rules! read_ints {
            (|$x:ident| $conv:expr) => {
                match src {
                    DType::I64 => tensor.storage::<i64>().iter().map(|&$x| $conv).collect(),
                    DType::I32 => tensor.storage::<i32>().iter().map(|&$x| $conv).collect(),
                    DType::I16 => tensor.storage::<i16>().iter().map(|&$x| $conv).collect(),
                    DType::I8 => tensor.storage::<i8>().iter().map(|&$x| $conv).collect(),
                    DType::U64 => tensor.storage::<u64>().iter().map(|&$x| $conv).collect(),
                    DType::U32 => tensor.storage::<u32>().iter().map(|&$x| $conv).collect(),
                    DType::U16 => tensor.storage::<u16>().iter().map(|&$x| $conv).collect(),
                    DType::U8 => tensor.storage::<u8>().iter().map(|&$x| $conv).collect(),
                    _ => panic!("int_into_float: unsupported source dtype {:?}", src),
                }
            };
        }

        match out_dtype {
            FloatDType::F64 => {
                let data: Vec<f64> = read_ints!(|x| x as f64);
                FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
            }
            FloatDType::F32 | FloatDType::Flex32 => {
                let data: Vec<f32> = read_ints!(|x| x as f32);
                FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
            }
            FloatDType::F16 => {
                let data: Vec<f16> = read_ints!(|x| f16::from_f32(x as f32));
                FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
            }
            FloatDType::BF16 => {
                let data: Vec<bf16> = read_ints!(|x| bf16::from_f32(x as f32));
                FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
            }
        }
    }

    fn int_swap_dims(tensor: IntTensor<Flex>, dim1: usize, dim2: usize) -> IntTensor<Flex> {
        tensor.transpose(dim1, dim2)
    }

    fn int_permute(tensor: IntTensor<Flex>, axes: &[usize]) -> IntTensor<Flex> {
        tensor.permute(axes)
    }

    fn int_flip(tensor: IntTensor<Flex>, axes: &[usize]) -> IntTensor<Flex> {
        crate::ops::flip::flip(tensor, axes)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        _device: &Device<Flex>,
        dtype: IntDType,
    ) -> IntTensor<Flex> {
        let mut seed = crate::backend::SEED.lock().unwrap();
        let mut rng = seed.take().unwrap_or_else(crate::backend::get_seeded_rng);
        let data = match dtype {
            IntDType::I64 => TensorData::random::<i64, _, _>(shape, distribution, &mut rng),
            IntDType::I32 => TensorData::random::<i32, _, _>(shape, distribution, &mut rng),
            IntDType::I16 => TensorData::random::<i16, _, _>(shape, distribution, &mut rng),
            IntDType::I8 => TensorData::random::<i8, _, _>(shape, distribution, &mut rng),
            IntDType::U64 => TensorData::random::<u64, _, _>(shape, distribution, &mut rng),
            IntDType::U32 => TensorData::random::<u32, _, _>(shape, distribution, &mut rng),
            IntDType::U16 => TensorData::random::<u16, _, _>(shape, distribution, &mut rng),
            IntDType::U8 => TensorData::random::<u8, _, _>(shape, distribution, &mut rng),
        };
        *seed = Some(rng);
        FlexTensor::from_data(data)
    }

    fn int_expand(tensor: IntTensor<Flex>, shape: Shape) -> IntTensor<Flex> {
        crate::ops::expand::expand(tensor, shape)
    }

    fn int_matmul(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        matmul::int_matmul(lhs, rhs)
    }

    fn int_sum(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        crate::ops::reduce::sum(tensor)
    }

    fn int_sum_dim(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::sum_dim(tensor, dim)
    }

    fn int_prod(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        crate::ops::reduce::prod(tensor)
    }

    fn int_prod_dim(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::prod_dim(tensor, dim)
    }

    fn int_mean_dim(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::mean_dim(tensor, dim)
    }

    fn int_cumsum(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        match tensor.dtype() {
            DType::I64 => crate::ops::cumulative::cumsum::<i64>(tensor, dim),
            DType::I32 => crate::ops::cumulative::cumsum::<i32>(tensor, dim),
            DType::I16 => crate::ops::cumulative::cumsum::<i16>(tensor, dim),
            DType::I8 => crate::ops::cumulative::cumsum::<i8>(tensor, dim),
            DType::U64 => crate::ops::cumulative::cumsum::<u64>(tensor, dim),
            DType::U32 => crate::ops::cumulative::cumsum::<u32>(tensor, dim),
            DType::U16 => crate::ops::cumulative::cumsum::<u16>(tensor, dim),
            DType::U8 => crate::ops::cumulative::cumsum::<u8>(tensor, dim),
            dt => panic!("int_cumsum: unsupported dtype {:?}", dt),
        }
    }

    fn int_cumprod(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        match tensor.dtype() {
            DType::I64 => crate::ops::cumulative::cumprod::<i64>(tensor, dim),
            DType::I32 => crate::ops::cumulative::cumprod::<i32>(tensor, dim),
            DType::I16 => crate::ops::cumulative::cumprod::<i16>(tensor, dim),
            DType::I8 => crate::ops::cumulative::cumprod::<i8>(tensor, dim),
            DType::U64 => crate::ops::cumulative::cumprod::<u64>(tensor, dim),
            DType::U32 => crate::ops::cumulative::cumprod::<u32>(tensor, dim),
            DType::U16 => crate::ops::cumulative::cumprod::<u16>(tensor, dim),
            DType::U8 => crate::ops::cumulative::cumprod::<u8>(tensor, dim),
            dt => panic!("int_cumprod: unsupported dtype {:?}", dt),
        }
    }

    fn int_cummin(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        match tensor.dtype() {
            DType::I64 => crate::ops::cumulative::cummin::<i64>(tensor, dim),
            DType::I32 => crate::ops::cumulative::cummin::<i32>(tensor, dim),
            DType::I16 => crate::ops::cumulative::cummin::<i16>(tensor, dim),
            DType::I8 => crate::ops::cumulative::cummin::<i8>(tensor, dim),
            DType::U64 => crate::ops::cumulative::cummin::<u64>(tensor, dim),
            DType::U32 => crate::ops::cumulative::cummin::<u32>(tensor, dim),
            DType::U16 => crate::ops::cumulative::cummin::<u16>(tensor, dim),
            DType::U8 => crate::ops::cumulative::cummin::<u8>(tensor, dim),
            dt => panic!("int_cummin: unsupported dtype {:?}", dt),
        }
    }

    fn int_cummax(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        match tensor.dtype() {
            DType::I64 => crate::ops::cumulative::cummax::<i64>(tensor, dim),
            DType::I32 => crate::ops::cumulative::cummax::<i32>(tensor, dim),
            DType::I16 => crate::ops::cumulative::cummax::<i16>(tensor, dim),
            DType::I8 => crate::ops::cumulative::cummax::<i8>(tensor, dim),
            DType::U64 => crate::ops::cumulative::cummax::<u64>(tensor, dim),
            DType::U32 => crate::ops::cumulative::cummax::<u32>(tensor, dim),
            DType::U16 => crate::ops::cumulative::cummax::<u16>(tensor, dim),
            DType::U8 => crate::ops::cumulative::cummax::<u8>(tensor, dim),
            dt => panic!("int_cummax: unsupported dtype {:?}", dt),
        }
    }

    fn int_argmax(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::argmax(tensor, dim)
    }

    fn int_argtopk(_tensor: IntTensor<Flex>, _dim: usize, _k: usize) -> IntTensor<Flex> {
        panic!("argtopk not implemented for flex")
    }

    fn int_argmin(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::argmin(tensor, dim)
    }

    fn int_abs(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        crate::ops::unary::int_abs(tensor)
    }

    fn bitwise_and(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a & b)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        if lhs.dtype() == DType::U64 {
            return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a & b);
        }
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a & b)
    }

    fn bitwise_or(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a | b)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        if lhs.dtype() == DType::U64 {
            return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a | b);
        }
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a | b)
    }

    fn bitwise_xor(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a ^ b)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        if lhs.dtype() == DType::U64 {
            return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a ^ b);
        }
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a ^ b)
    }

    fn bitwise_not(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        // Use scalar op with dummy value, only applying NOT to lhs
        int_scalar_op(tensor, 0, |a, _| !a)
    }

    // Shift amounts masked to type width via wrapping_shl/wrapping_shr.
    fn bitwise_left_shift(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a.wrapping_shl(b as u32))
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a.wrapping_shl(b as u32))
    }

    fn bitwise_right_shift(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a.wrapping_shr(b as u32))
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a.wrapping_shr(b as u32))
    }

    fn int_cast(tensor: IntTensor<Flex>, dtype: IntDType) -> IntTensor<Flex> {
        let target_dtype: DType = dtype.into();

        // If already the target dtype, return as-is
        if tensor.dtype() == target_dtype {
            return tensor;
        }

        // Make contiguous for easier iteration
        let tensor = tensor.to_contiguous();
        let shape = tensor.layout().shape().clone();

        // Helper macro to convert between types
        macro_rules! cast_impl {
            ($src_type:ty, $dst_type:ty, $dst_dtype:expr) => {{
                let src: &[$src_type] = tensor.storage();
                let dst: Vec<$dst_type> = src.iter().map(|&x| x as $dst_type).collect();
                FlexTensor::new(
                    Bytes::from_elems(dst),
                    Layout::contiguous(shape),
                    $dst_dtype,
                )
            }};
        }

        // Match source dtype to target dtype
        match (tensor.dtype(), target_dtype) {
            // From I64
            (DType::I64, DType::I32) => cast_impl!(i64, i32, DType::I32),
            (DType::I64, DType::I16) => cast_impl!(i64, i16, DType::I16),
            (DType::I64, DType::I8) => cast_impl!(i64, i8, DType::I8),
            (DType::I64, DType::U64) => cast_impl!(i64, u64, DType::U64),
            (DType::I64, DType::U32) => cast_impl!(i64, u32, DType::U32),
            (DType::I64, DType::U16) => cast_impl!(i64, u16, DType::U16),
            (DType::I64, DType::U8) => cast_impl!(i64, u8, DType::U8),

            // From I32
            (DType::I32, DType::I64) => cast_impl!(i32, i64, DType::I64),
            (DType::I32, DType::I16) => cast_impl!(i32, i16, DType::I16),
            (DType::I32, DType::I8) => cast_impl!(i32, i8, DType::I8),
            (DType::I32, DType::U64) => cast_impl!(i32, u64, DType::U64),
            (DType::I32, DType::U32) => cast_impl!(i32, u32, DType::U32),
            (DType::I32, DType::U16) => cast_impl!(i32, u16, DType::U16),
            (DType::I32, DType::U8) => cast_impl!(i32, u8, DType::U8),

            // From I16
            (DType::I16, DType::I64) => cast_impl!(i16, i64, DType::I64),
            (DType::I16, DType::I32) => cast_impl!(i16, i32, DType::I32),
            (DType::I16, DType::I8) => cast_impl!(i16, i8, DType::I8),
            (DType::I16, DType::U64) => cast_impl!(i16, u64, DType::U64),
            (DType::I16, DType::U32) => cast_impl!(i16, u32, DType::U32),
            (DType::I16, DType::U16) => cast_impl!(i16, u16, DType::U16),
            (DType::I16, DType::U8) => cast_impl!(i16, u8, DType::U8),

            // From I8
            (DType::I8, DType::I64) => cast_impl!(i8, i64, DType::I64),
            (DType::I8, DType::I32) => cast_impl!(i8, i32, DType::I32),
            (DType::I8, DType::I16) => cast_impl!(i8, i16, DType::I16),
            (DType::I8, DType::U64) => cast_impl!(i8, u64, DType::U64),
            (DType::I8, DType::U32) => cast_impl!(i8, u32, DType::U32),
            (DType::I8, DType::U16) => cast_impl!(i8, u16, DType::U16),
            (DType::I8, DType::U8) => cast_impl!(i8, u8, DType::U8),

            // From U64
            (DType::U64, DType::I64) => cast_impl!(u64, i64, DType::I64),
            (DType::U64, DType::I32) => cast_impl!(u64, i32, DType::I32),
            (DType::U64, DType::I16) => cast_impl!(u64, i16, DType::I16),
            (DType::U64, DType::I8) => cast_impl!(u64, i8, DType::I8),
            (DType::U64, DType::U32) => cast_impl!(u64, u32, DType::U32),
            (DType::U64, DType::U16) => cast_impl!(u64, u16, DType::U16),
            (DType::U64, DType::U8) => cast_impl!(u64, u8, DType::U8),

            // From U32
            (DType::U32, DType::I64) => cast_impl!(u32, i64, DType::I64),
            (DType::U32, DType::I32) => cast_impl!(u32, i32, DType::I32),
            (DType::U32, DType::I16) => cast_impl!(u32, i16, DType::I16),
            (DType::U32, DType::I8) => cast_impl!(u32, i8, DType::I8),
            (DType::U32, DType::U64) => cast_impl!(u32, u64, DType::U64),
            (DType::U32, DType::U16) => cast_impl!(u32, u16, DType::U16),
            (DType::U32, DType::U8) => cast_impl!(u32, u8, DType::U8),

            // From U16
            (DType::U16, DType::I64) => cast_impl!(u16, i64, DType::I64),
            (DType::U16, DType::I32) => cast_impl!(u16, i32, DType::I32),
            (DType::U16, DType::I16) => cast_impl!(u16, i16, DType::I16),
            (DType::U16, DType::I8) => cast_impl!(u16, i8, DType::I8),
            (DType::U16, DType::U64) => cast_impl!(u16, u64, DType::U64),
            (DType::U16, DType::U32) => cast_impl!(u16, u32, DType::U32),
            (DType::U16, DType::U8) => cast_impl!(u16, u8, DType::U8),

            // From U8
            (DType::U8, DType::I64) => cast_impl!(u8, i64, DType::I64),
            (DType::U8, DType::I32) => cast_impl!(u8, i32, DType::I32),
            (DType::U8, DType::I16) => cast_impl!(u8, i16, DType::I16),
            (DType::U8, DType::I8) => cast_impl!(u8, i8, DType::I8),
            (DType::U8, DType::U64) => cast_impl!(u8, u64, DType::U64),
            (DType::U8, DType::U32) => cast_impl!(u8, u32, DType::U32),
            (DType::U8, DType::U16) => cast_impl!(u8, u16, DType::U16),

            _ => panic!(
                "int_cast: unsupported conversion from {:?} to {:?}",
                tensor.dtype(),
                target_dtype
            ),
        }
    }

    fn int_unfold(
        tensor: IntTensor<Flex>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Flex> {
        crate::ops::unfold::unfold_int(tensor, dim, size, step)
    }

    fn int_neg(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        int_scalar_op(tensor, 0i64, |a, _| a.wrapping_neg())
    }

    fn int_clamp(tensor: IntTensor<Flex>, min: Scalar, max: Scalar) -> IntTensor<Flex> {
        if tensor.dtype() == DType::U64 {
            let min_val = min.to_u64().unwrap();
            let max_val = max.to_u64().unwrap();
            return scalar_op_typed(tensor, 0u64, move |x: u64, _| x.clamp(min_val, max_val));
        }
        let min_val = min.to_i64().unwrap();
        let max_val = max.to_i64().unwrap();
        int_scalar_op(tensor, 0i64, move |x, _| x.clamp(min_val, max_val))
    }

    fn int_clamp_min(tensor: IntTensor<Flex>, min: Scalar) -> IntTensor<Flex> {
        if tensor.dtype() == DType::U64 {
            let min_val = min.to_u64().unwrap();
            return scalar_op_typed(tensor, 0u64, move |x: u64, _| x.max(min_val));
        }
        let min_val = min.to_i64().unwrap();
        int_scalar_op(tensor, 0i64, move |x, _| x.max(min_val))
    }

    fn int_clamp_max(tensor: IntTensor<Flex>, max: Scalar) -> IntTensor<Flex> {
        if tensor.dtype() == DType::U64 {
            let max_val = max.to_u64().unwrap();
            return scalar_op_typed(tensor, 0u64, move |x: u64, _| x.min(max_val));
        }
        let max_val = max.to_i64().unwrap();
        int_scalar_op(tensor, 0i64, move |x, _| x.min(max_val))
    }

    fn int_sign(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        if tensor.dtype() == DType::U64 {
            return scalar_op_typed(tensor, 0u64, |x: u64, _| if x > 0 { 1 } else { 0 });
        }
        int_scalar_op(tensor, 0i64, |x, _| {
            if x > 0 {
                1
            } else if x < 0 {
                -1
            } else {
                0
            }
        })
    }

    fn int_mean(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        let n = tensor.layout().num_elements();
        assert!(n > 0, "int_mean: cannot take mean of empty tensor");
        let dtype = tensor.dtype();
        let sum_result = crate::ops::reduce::sum(tensor);
        // Compute in i64 to avoid truncation of n for small int types
        macro_rules! compute_mean {
            ($ty:ty) => {{
                let data: &[$ty] = sum_result.storage();
                let mean_val = (data[0] as i64 / n as i64) as $ty;
                FlexTensor::new(
                    Bytes::from_elems(alloc::vec![mean_val]),
                    Layout::contiguous(Shape::from(alloc::vec![1])),
                    dtype,
                )
            }};
        }
        match dtype {
            DType::I64 => compute_mean!(i64),
            DType::I32 => compute_mean!(i32),
            DType::I16 => compute_mean!(i16),
            DType::I8 => compute_mean!(i8),
            other => panic!("int_mean: unsupported dtype {:?}", other),
        }
    }

    fn int_max(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        crate::ops::reduce::max(tensor)
    }

    fn int_max_dim(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::max_dim(tensor, dim)
    }

    fn int_min(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        crate::ops::reduce::min(tensor)
    }

    fn int_min_dim(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::min_dim(tensor, dim)
    }

    fn int_max_dim_with_indices(
        tensor: IntTensor<Flex>,
        dim: usize,
    ) -> (IntTensor<Flex>, IntTensor<Flex>) {
        crate::ops::reduce::max_dim_with_indices(tensor, dim)
    }

    fn int_min_dim_with_indices(
        tensor: IntTensor<Flex>,
        dim: usize,
    ) -> (IntTensor<Flex>, IntTensor<Flex>) {
        crate::ops::reduce::min_dim_with_indices(tensor, dim)
    }

    fn int_any(tensor: IntTensor<Flex>, out_dtype: burn_std::BoolDType) -> BoolTensor<Flex> {
        crate::ops::comparison::any_int(tensor, out_dtype)
    }

    fn int_any_dim(
        tensor: IntTensor<Flex>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::any_int_dim(tensor, dim, out_dtype)
    }

    fn int_all(tensor: IntTensor<Flex>, out_dtype: burn_std::BoolDType) -> BoolTensor<Flex> {
        crate::ops::comparison::all_int(tensor, out_dtype)
    }

    fn int_all_dim(
        tensor: IntTensor<Flex>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::all_int_dim(tensor, dim, out_dtype)
    }

    fn int_powi(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a.wrapping_pow(b as u32))
    }

    fn int_zeros(shape: Shape, _device: &Device<Flex>, dtype: IntDType) -> IntTensor<Flex> {
        FlexTensor::zeros(shape, dtype.into())
    }

    fn int_ones(shape: Shape, _device: &Device<Flex>, dtype: IntDType) -> IntTensor<Flex> {
        let dt: DType = dtype.into();
        match dt {
            DType::I64 => FlexTensor::filled_typed(shape, dt, 1i64),
            DType::I32 => FlexTensor::filled_typed(shape, dt, 1i32),
            DType::I16 => FlexTensor::filled_typed(shape, dt, 1i16),
            DType::I8 => FlexTensor::filled_typed(shape, dt, 1i8),
            DType::U64 => FlexTensor::filled_typed(shape, dt, 1u64),
            DType::U32 => FlexTensor::filled_typed(shape, dt, 1u32),
            DType::U16 => FlexTensor::filled_typed(shape, dt, 1u16),
            DType::U8 => FlexTensor::filled_typed(shape, dt, 1u8),
            _ => unreachable!(),
        }
    }

    fn int_full(
        shape: Shape,
        fill_value: burn_backend::Scalar,
        _device: &Device<Flex>,
        dtype: IntDType,
    ) -> IntTensor<Flex> {
        let dt: DType = dtype.into();
        let v = fill_value.to_i64().unwrap();
        match dt {
            DType::I64 => FlexTensor::filled_typed(shape, dt, v),
            DType::I32 => FlexTensor::filled_typed(shape, dt, v as i32),
            DType::I16 => FlexTensor::filled_typed(shape, dt, v as i16),
            DType::I8 => FlexTensor::filled_typed(shape, dt, v as i8),
            DType::U64 => FlexTensor::filled_typed(shape, dt, v as u64),
            DType::U32 => FlexTensor::filled_typed(shape, dt, v as u32),
            DType::U16 => FlexTensor::filled_typed(shape, dt, v as u16),
            DType::U8 => FlexTensor::filled_typed(shape, dt, v as u8),
            _ => unreachable!(),
        }
    }

    fn int_transpose(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        let ndims = tensor.layout().num_dims();
        if ndims < 2 {
            return tensor;
        }
        tensor.transpose(ndims - 2, ndims - 1)
    }

    fn int_repeat_dim(tensor: IntTensor<Flex>, dim: usize, times: usize) -> IntTensor<Flex> {
        crate::ops::repeat_dim::repeat_dim(tensor, dim, times)
    }

    fn int_not_equal(
        lhs: IntTensor<Flex>,
        rhs: IntTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::int_not_equal(lhs, rhs, out_dtype)
    }

    fn int_not_equal_elem(
        lhs: IntTensor<Flex>,
        rhs: burn_backend::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
        crate::ops::comparison::int_not_equal_elem(lhs, i, u, out_dtype)
    }

    fn int_sort(tensor: IntTensor<Flex>, dim: usize, descending: bool) -> IntTensor<Flex> {
        crate::ops::sort::sort(tensor, dim, descending)
    }

    fn int_sort_with_indices(
        tensor: IntTensor<Flex>,
        dim: usize,
        descending: bool,
    ) -> (IntTensor<Flex>, IntTensor<Flex>) {
        crate::ops::sort::sort_with_indices(tensor, dim, descending)
    }

    fn int_argsort(tensor: IntTensor<Flex>, dim: usize, descending: bool) -> IntTensor<Flex> {
        crate::ops::sort::argsort(tensor, dim, descending)
    }

    fn int_powi_scalar(lhs: IntTensor<Flex>, rhs: burn_backend::Scalar) -> IntTensor<Flex> {
        use num_traits::ToPrimitive;
        match rhs.to_i64().unwrap() {
            0 => Self::int_ones(lhs.shape(), &Default::default(), lhs.dtype().into()),
            1 => lhs,
            2 => Self::int_mul(lhs.clone(), lhs),
            _ => Self::int_powi_scalar_impl(lhs, rhs),
        }
    }

    fn int_powi_scalar_impl(lhs: IntTensor<Flex>, rhs: burn_backend::Scalar) -> IntTensor<Flex> {
        use num_traits::ToPrimitive;
        let exp = rhs.to_i64().unwrap() as u32;
        if lhs.dtype() == DType::U64 {
            return scalar_op_typed(lhs, exp as u64, move |x: u64, _| x.wrapping_pow(exp));
        }
        int_scalar_op(lhs, exp as i64, move |x, _| x.wrapping_pow(exp))
    }

    fn int_max_abs(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        let abs = Self::int_abs(tensor);
        crate::ops::reduce::max(abs)
    }

    fn int_max_abs_dim(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        let abs = Self::int_abs(tensor);
        crate::ops::reduce::max_dim(abs, dim)
    }

    fn int_arange(
        range: core::ops::Range<i64>,
        _device: &Device<Flex>,
        dtype: IntDType,
    ) -> IntTensor<Flex> {
        Self::int_arange_step(range, 1, &Default::default(), dtype)
    }

    fn int_arange_step(
        range: core::ops::Range<i64>,
        step: usize,
        _device: &Device<Flex>,
        dtype: IntDType,
    ) -> IntTensor<Flex> {
        let dt: DType = dtype.into();

        macro_rules! arange_typed {
            ($ty:ty) => {{
                let data: Vec<$ty> = range.step_by(step).map(|v| v as $ty).collect();
                let shape = Shape::from(alloc::vec![data.len()]);
                FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), dt)
            }};
        }

        match dt {
            DType::I64 => arange_typed!(i64),
            DType::I32 => arange_typed!(i32),
            DType::I16 => arange_typed!(i16),
            DType::I8 => arange_typed!(i8),
            DType::U64 => arange_typed!(u64),
            DType::U32 => arange_typed!(u32),
            DType::U16 => arange_typed!(u16),
            DType::U8 => arange_typed!(u8),
            _ => unreachable!(),
        }
    }
}

// Tests kept here exercise flex-specific behavior: dtype storage
// selection for every int width (I16/I32/U8/U16/U32/I64/U64), and edge
// cases of the dtype-specific kernels (u64 wrap, i64::MIN abs/neg, bit
// shift at width). Plain int arithmetic, scalar ops, bool->int cast
// smokes, and negative-stride (flipped/transposed) variants have been
// migrated to burn-backend-tests so they run against every backend.
// When adding new tests, keep them here only if they probe flex dtype
// storage; otherwise add them to
// crates/burn-backend-tests/tests/tensor/int/ops/.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use burn_backend::TensorData;
    use burn_backend::ops::IntTensorOps;

    use crate::Flex;
    use crate::FlexTensor;

    #[test]
    fn test_u64_div_large_values() {
        let a = FlexTensor::from_data(TensorData::new(vec![u64::MAX], [1]));
        let b = FlexTensor::from_data(TensorData::new(vec![2u64], [1]));
        let result = Flex::int_div(a, b);
        let values: Vec<u64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], u64::MAX / 2);
    }

    #[test]
    fn test_u64_remainder_large_values() {
        let a = FlexTensor::from_data(TensorData::new(vec![u64::MAX], [1]));
        let b = FlexTensor::from_data(TensorData::new(vec![2u64], [1]));
        let result = Flex::int_remainder(a, b);
        let values: Vec<u64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], u64::MAX % 2);
    }

    #[test]
    fn test_int_abs_min_value() {
        // i64::MIN.abs() panics in debug; wrapping_abs returns MIN (matches PyTorch)
        let a = FlexTensor::from_data(TensorData::new(vec![i64::MIN], [1]));
        let result = Flex::int_abs(a);
        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], i64::MIN.wrapping_abs());
    }

    #[test]
    fn test_int_neg_min_value() {
        // i64::MIN negation panics in debug; wrapping_neg returns MIN (matches PyTorch)
        let a = FlexTensor::from_data(TensorData::new(vec![i64::MIN], [1]));
        let result = Flex::int_neg(a);
        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], i64::MIN.wrapping_neg());
    }

    #[test]
    fn test_int_shift_large_amount() {
        // Shift by >= bit width panics without wrapping; should not crash
        let a = FlexTensor::from_data(TensorData::new(vec![1i64], [1]));
        let b = FlexTensor::from_data(TensorData::new(vec![64i64], [1]));
        let _left = Flex::bitwise_left_shift(a.clone(), b.clone());
        let _right = Flex::bitwise_right_shift(a, b);
    }

    #[test]
    fn test_int_into_float_f64() {
        use burn_backend::ops::IntTensorOps;
        use burn_std::FloatDType;

        let t = FlexTensor::from_data(TensorData::new(vec![1i64, 2, -3], [3]));
        let result = Flex::int_into_float(t, FloatDType::F64);
        assert_eq!(result.dtype(), burn_backend::DType::F64);
        let data: Vec<f64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0f64, 2.0, -3.0]);
    }

    #[test]
    fn test_u64_add_scalar_large() {
        let t = FlexTensor::from_data(TensorData::new(vec![1u64, 2, 3], [3]));
        let big: u64 = (i64::MAX as u64) + 100;
        let result = Flex::int_add_scalar(t, burn_backend::Scalar::from(big));
        let data: Vec<u64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![big + 1, big + 2, big + 3]);
    }

    #[test]
    fn test_u64_greater_elem_large() {
        let big: u64 = (i64::MAX as u64) + 100;
        let t = FlexTensor::from_data(TensorData::new(vec![big, big + 1, big - 1], [3]));
        let result = Flex::int_greater_elem(
            t,
            burn_backend::Scalar::from(big),
            burn_std::BoolStore::Native,
        );
        let data: Vec<bool> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![false, true, false]);
    }

    #[test]
    fn test_int_mask_fill_i32() {
        let t = FlexTensor::from_data(TensorData::new(vec![1i32, 2, 3, 4], [4]));
        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, true, false], [4]));
        let result = Flex::int_mask_fill(t, mask, burn_backend::Scalar::from(0i64));
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0, 2, 0, 4]);
    }

    #[test]
    fn test_int_mask_fill_i16() {
        let t = FlexTensor::from_data(TensorData::new(vec![10i16, 20, 30, 40], [4]));
        let mask = FlexTensor::from_data(TensorData::new(vec![false, true, false, true], [4]));
        let result = Flex::int_mask_fill(t, mask, burn_backend::Scalar::from(-1i64));
        let data: Vec<i16> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![10, -1, 30, -1]);
    }

    #[test]
    fn test_int_mask_fill_u8() {
        let t = FlexTensor::from_data(TensorData::new(vec![1u8, 2, 3, 4], [4]));
        let mask = FlexTensor::from_data(TensorData::new(vec![true, true, false, false], [4]));
        let result = Flex::int_mask_fill(t, mask, burn_backend::Scalar::from(255i64));
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![255, 255, 3, 4]);
    }

    #[test]
    fn test_int_mask_fill_u32() {
        let t = FlexTensor::from_data(TensorData::new(vec![100u32, 200, 300], [3]));
        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, true], [3]));
        let result = Flex::int_mask_fill(t, mask, burn_backend::Scalar::from(0i64));
        let data: Vec<u32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0, 200, 0]);
    }

    #[test]
    fn test_int_mask_where_i32() {
        let t = FlexTensor::from_data(TensorData::new(vec![1i32, 2, 3, 4], [4]));
        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, true, false], [4]));
        let v = FlexTensor::from_data(TensorData::new(vec![10i32, 20, 30, 40], [4]));
        let result = Flex::int_mask_where(t, mask, v);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![10, 2, 30, 4]);
    }

    #[test]
    fn test_int_mask_where_u8() {
        let t = FlexTensor::from_data(TensorData::new(vec![1u8, 2, 3, 4], [4]));
        let mask = FlexTensor::from_data(TensorData::new(vec![false, true, false, true], [4]));
        let v = FlexTensor::from_data(TensorData::new(vec![10u8, 20, 30, 40], [4]));
        let result = Flex::int_mask_where(t, mask, v);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1, 20, 3, 40]);
    }

    #[test]
    fn test_int_gather_i32() {
        let t = FlexTensor::from_data(TensorData::new(vec![10i32, 20, 30, 40, 50, 60], [2, 3]));
        let indices = FlexTensor::from_data(TensorData::new(vec![2i64, 0, 1, 2], [2, 2]));
        let result = Flex::int_gather(1, t, indices);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![30, 10, 50, 60]);
    }

    #[test]
    fn test_int_select_u16() {
        let t = FlexTensor::from_data(TensorData::new(vec![10u16, 20, 30, 40, 50, 60], [2, 3]));
        let indices = FlexTensor::from_data(TensorData::new(vec![0i64, 1], [2]));
        let result = Flex::int_select(t, 1, indices);
        let data: Vec<u16> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![10, 20, 40, 50]);
    }

    #[test]
    fn test_int_cumsum_i32() {
        let t = FlexTensor::from_data(TensorData::new(vec![1i32, 2, 3, 4], [4]));
        let result = Flex::int_cumsum(t, 0);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1, 3, 6, 10]);
    }

    #[test]
    fn test_int_cumprod_u8() {
        let t = FlexTensor::from_data(TensorData::new(vec![1u8, 2, 3, 4], [4]));
        let result = Flex::int_cumprod(t, 0);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1, 2, 6, 24]);
    }

    #[test]
    fn test_int_cummin_i32() {
        let t = FlexTensor::from_data(TensorData::new(vec![3i32, 1, 4, 1, 5], [5]));
        let result = Flex::int_cummin(t, 0);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3, 1, 1, 1, 1]);
    }

    #[test]
    fn test_int_cummax_u16() {
        let t = FlexTensor::from_data(TensorData::new(vec![3u16, 1, 4, 1, 5], [5]));
        let result = Flex::int_cummax(t, 0);
        let data: Vec<u16> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3, 3, 4, 4, 5]);
    }

    #[test]
    fn test_int_scatter_add_i32() {
        let t = FlexTensor::from_data(TensorData::new(vec![0i32, 0, 0], [1, 3]));
        let indices = FlexTensor::from_data(TensorData::new(vec![0i64, 2, 1], [1, 3]));
        let values = FlexTensor::from_data(TensorData::new(vec![10i32, 20, 30], [1, 3]));
        let result = Flex::int_scatter_add(1, t, indices, values);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![10, 30, 20]);
    }

    #[test]
    fn test_int_select_add_u8() {
        let t = FlexTensor::from_data(TensorData::new(vec![1u8, 2, 3], [3]));
        let indices = FlexTensor::from_data(TensorData::new(vec![0i64, 2], [2]));
        let values = FlexTensor::from_data(TensorData::new(vec![10u8, 20], [2]));
        let result = Flex::int_select_add(t, 0, indices, values);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![11, 2, 23]);
    }

    #[test]
    fn test_int_random_i32() {
        use burn_backend::{DType, Distribution, ops::IntTensorOps};
        use burn_std::{IntDType, Shape};

        let shape = Shape::from(vec![100]);
        let dist = Distribution::Uniform(0.0, 10.0);
        let device = crate::FlexDevice;
        let t = Flex::int_random(shape, dist, &device, IntDType::I32);
        assert_eq!(t.dtype(), DType::I32);
        let data: Vec<i32> = t.into_data().to_vec().unwrap();
        assert!(data.iter().all(|&v| (0..=10).contains(&v)));
    }

    #[test]
    fn test_int_random_u8() {
        use burn_backend::{DType, Distribution, ops::IntTensorOps};
        use burn_std::{IntDType, Shape};

        let shape = Shape::from(vec![50]);
        let dist = Distribution::Uniform(0.0, 100.0);
        let device = crate::FlexDevice;
        let t = Flex::int_random(shape, dist, &device, IntDType::U8);
        assert_eq!(t.dtype(), DType::U8);
    }

    #[test]
    fn test_int_mean_i32() {
        use burn_backend::{DType, ops::IntTensorOps};

        let t = FlexTensor::from_data(TensorData::new(vec![10i32, 20, 30], [3]));
        let result = Flex::int_mean(t);
        assert_eq!(result.dtype(), DType::I32);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![20]); // (10 + 20 + 30) / 3 = 20
    }
}
