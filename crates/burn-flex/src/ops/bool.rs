//! Bool tensor operations for the Flex backend.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{
    DType, ExecutionError, TensorData,
    ops::{BoolTensorOps, IntTensorOps},
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_std::{Bytes, FloatDType, IntDType, Shape, Slice, bf16, f16};

use crate::{Flex, FlexTensor, Layout};

impl BoolTensorOps<Flex> for Flex {
    fn bool_from_data(data: TensorData, _device: &Device<Flex>) -> BoolTensor<Flex> {
        FlexTensor::from_data(data)
    }

    async fn bool_into_data(tensor: BoolTensor<Flex>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn bool_device(_tensor: &BoolTensor<Flex>) -> Device<Flex> {
        Default::default()
    }

    fn bool_to_device(tensor: BoolTensor<Flex>, _device: &Device<Flex>) -> BoolTensor<Flex> {
        tensor
    }

    fn bool_cat(tensors: Vec<BoolTensor<Flex>>, dim: usize) -> BoolTensor<Flex> {
        crate::ops::cat::cat(tensors, dim)
    }

    fn bool_reshape(tensor: BoolTensor<Flex>, shape: Shape) -> BoolTensor<Flex> {
        tensor.reshape(shape)
    }

    fn bool_slice(tensor: BoolTensor<Flex>, slices: &[Slice]) -> BoolTensor<Flex> {
        crate::ops::slice::slice(tensor, slices)
    }

    fn bool_empty(
        shape: Shape,
        _device: &Device<Flex>,
        dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        FlexTensor::empty(shape, DType::from(dtype))
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Flex>,
        slices: &[Slice],
        value: BoolTensor<Flex>,
    ) -> BoolTensor<Flex> {
        crate::ops::slice::slice_assign(tensor, slices, value)
    }

    fn bool_into_int(tensor: BoolTensor<Flex>, out_dtype: burn_std::IntDType) -> IntTensor<Flex> {
        let tensor = tensor.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let out_dt = DType::from(out_dtype);
        let bools = tensor.bytes();

        macro_rules! convert {
            ($int_ty:ty) => {{
                let data: Vec<$int_ty> =
                    bools.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();
                FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
            }};
        }

        match out_dtype {
            IntDType::I64 => convert!(i64),
            IntDType::I32 => convert!(i32),
            IntDType::I16 => convert!(i16),
            IntDType::I8 => convert!(i8),
            IntDType::U64 => convert!(u64),
            IntDType::U32 => convert!(u32),
            IntDType::U16 => convert!(u16),
            IntDType::U8 => convert!(u8),
        }
    }

    fn bool_into_float(
        tensor: BoolTensor<Flex>,
        out_dtype: burn_std::FloatDType,
    ) -> FloatTensor<Flex> {
        let tensor = tensor.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let out_dt = DType::from(out_dtype);
        let bools = tensor.bytes();

        match out_dtype {
            FloatDType::F64 => {
                let data: Vec<f64> = bools
                    .iter()
                    .map(|&x| if x != 0 { 1.0 } else { 0.0 })
                    .collect();
                FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
            }
            FloatDType::F32 | FloatDType::Flex32 => {
                let data: Vec<f32> = bools
                    .iter()
                    .map(|&x| if x != 0 { 1.0 } else { 0.0 })
                    .collect();
                FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
            }
            FloatDType::F16 => {
                let one = f16::from_f32(1.0);
                let zero = f16::from_f32(0.0);
                let data: Vec<f16> = bools
                    .iter()
                    .map(|&x| if x != 0 { one } else { zero })
                    .collect();
                FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
            }
            FloatDType::BF16 => {
                let one = bf16::from_f32(1.0);
                let zero = bf16::from_f32(0.0);
                let data: Vec<bf16> = bools
                    .iter()
                    .map(|&x| if x != 0 { one } else { zero })
                    .collect();
                FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
            }
        }
    }

    fn bool_swap_dims(tensor: BoolTensor<Flex>, dim1: usize, dim2: usize) -> BoolTensor<Flex> {
        tensor.transpose(dim1, dim2)
    }

    fn bool_permute(tensor: BoolTensor<Flex>, axes: &[usize]) -> BoolTensor<Flex> {
        tensor.permute(axes)
    }

    fn bool_flip(tensor: BoolTensor<Flex>, axes: &[usize]) -> BoolTensor<Flex> {
        crate::ops::flip::flip(tensor, axes)
    }

    fn bool_equal(lhs: BoolTensor<Flex>, rhs: BoolTensor<Flex>) -> BoolTensor<Flex> {
        use crate::strided_index::StridedIter;

        // Broadcast to a common shape before comparing. The contiguous fast
        // path below uses `zip`, which silently truncates to the shorter
        // operand; and the output shape is taken from lhs, so mismatched
        // operands would otherwise produce a result vec shorter than the
        // output layout claims.
        let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

        let out_dtype = burn_std::BoolDType::from(lhs.dtype());
        let shape = lhs.layout().shape().clone();
        let lhs_storage: &[u8] = lhs.bytes();
        let rhs_storage: &[u8] = rhs.bytes();

        let result: Vec<u8> = match (
            lhs.layout().contiguous_offsets(),
            rhs.layout().contiguous_offsets(),
        ) {
            (Some((l_start, l_end)), Some((r_start, r_end))) => {
                let l_slice = &lhs_storage[l_start..l_end];
                let r_slice = &rhs_storage[r_start..r_end];
                l_slice
                    .iter()
                    .zip(r_slice)
                    .map(|(&a, &b)| (a == b) as u8)
                    .collect()
            }
            _ => {
                let lhs_iter = StridedIter::new(lhs.layout());
                let rhs_iter = StridedIter::new(rhs.layout());
                lhs_iter
                    .zip(rhs_iter)
                    .map(|(li, ri)| (lhs_storage[li] == rhs_storage[ri]) as u8)
                    .collect()
            }
        };

        crate::ops::comparison::make_bool_tensor(result, shape, out_dtype)
    }

    fn bool_not(mut tensor: BoolTensor<Flex>) -> BoolTensor<Flex> {
        use crate::strided_index::StridedIter;

        debug_assert!(
            matches!(
                tensor.dtype(),
                DType::Bool(burn_std::BoolStore::Native | burn_std::BoolStore::U8)
            ),
            "bool_not: only Bool(Native) and Bool(U8) are supported, got {:?}",
            tensor.dtype()
        );

        // Fast path: in-place for unique, contiguous tensors at offset 0. This
        // preserves the input tensor's dtype tag implicitly (the in-place SIMD
        // ops flip bytes without touching the dtype tag).
        if tensor.is_unique()
            && tensor.layout().is_contiguous()
            && tensor.layout().start_offset() == 0
        {
            let storage = tensor.storage_mut::<u8>();
            crate::simd::bool_not_inplace_u8(storage);
            return tensor;
        }

        // Allocating path for shared, non-contiguous, or offset tensors:
        // preserve the input's bool dtype for the new tensor.
        let out_dtype = burn_std::BoolDType::from(tensor.dtype());
        let shape = tensor.layout().shape().clone();
        let storage: &[u8] = tensor.bytes();

        let result: Vec<u8> = match tensor.layout().contiguous_offsets() {
            Some((start, end)) => {
                let slice = &storage[start..end];
                let mut out = vec![0u8; slice.len()];
                crate::simd::bool_not_u8(slice, &mut out);
                out
            }
            None => StridedIter::new(tensor.layout())
                .map(|idx| (storage[idx] == 0) as u8)
                .collect(),
        };

        crate::ops::comparison::make_bool_tensor(result, shape, out_dtype)
    }

    fn bool_and(lhs: BoolTensor<Flex>, rhs: BoolTensor<Flex>) -> BoolTensor<Flex> {
        bool_binary_op_simd(lhs, rhs, BoolBinaryOp::And)
    }

    fn bool_or(lhs: BoolTensor<Flex>, rhs: BoolTensor<Flex>) -> BoolTensor<Flex> {
        bool_binary_op_simd(lhs, rhs, BoolBinaryOp::Or)
    }

    fn bool_xor(lhs: BoolTensor<Flex>, rhs: BoolTensor<Flex>) -> BoolTensor<Flex> {
        bool_binary_op_simd(lhs, rhs, BoolBinaryOp::Xor)
    }

    fn bool_expand(tensor: BoolTensor<Flex>, shape: Shape) -> BoolTensor<Flex> {
        crate::ops::expand::expand(tensor, shape)
    }

    // Missing methods
    fn bool_zeros(
        shape: Shape,
        device: &Device<Flex>,
        dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        Self::bool_empty(shape, device, dtype)
    }

    fn bool_ones(
        shape: Shape,
        _device: &Device<Flex>,
        dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        let num_elements = shape.num_elements();
        let data = vec![1u8; num_elements];
        crate::ops::comparison::make_bool_tensor(data, shape, dtype)
    }

    fn bool_mask_where(
        tensor: BoolTensor<Flex>,
        mask: BoolTensor<Flex>,
        value: BoolTensor<Flex>,
    ) -> BoolTensor<Flex> {
        crate::ops::mask::mask_where_bool(tensor, mask, value)
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Flex>,
        mask: BoolTensor<Flex>,
        value: burn_backend::Scalar,
    ) -> BoolTensor<Flex> {
        let value: bool = value.elem();
        crate::ops::mask::mask_fill_bool(tensor, mask, value)
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Flex>,
        indices: IntTensor<Flex>,
    ) -> BoolTensor<Flex> {
        crate::ops::gather_scatter::gather_bool(tensor, dim, indices)
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Flex>,
        indices: IntTensor<Flex>,
        value: BoolTensor<Flex>,
    ) -> BoolTensor<Flex> {
        crate::ops::gather_scatter::scatter_or(tensor, dim, indices, value)
    }

    fn bool_equal_elem(lhs: BoolTensor<Flex>, rhs: burn_backend::Scalar) -> BoolTensor<Flex> {
        use crate::strided_index::StridedIter;

        let out_dtype = burn_std::BoolDType::from(lhs.dtype());
        let shape = lhs.layout().shape().clone();
        let storage: &[u8] = lhs.bytes();
        let rhs_bool: bool = rhs.elem();
        let rhs_val = rhs_bool as u8;

        let result: Vec<u8> = match lhs.layout().contiguous_offsets() {
            Some((start, end)) => storage[start..end]
                .iter()
                .map(|&v| (v == rhs_val) as u8)
                .collect(),
            None => StridedIter::new(lhs.layout())
                .map(|idx| (storage[idx] == rhs_val) as u8)
                .collect(),
        };

        crate::ops::comparison::make_bool_tensor(result, shape, out_dtype)
    }

    fn bool_unfold(
        tensor: BoolTensor<Flex>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Flex> {
        crate::ops::unfold::unfold_bool(tensor, dim, size, step)
    }

    fn bool_not_equal(lhs: BoolTensor<Flex>, rhs: BoolTensor<Flex>) -> BoolTensor<Flex> {
        let out_dtype = burn_std::BoolDType::from(lhs.dtype());
        crate::ops::comparison::bool_not_equal(lhs, rhs, out_dtype)
    }

    fn bool_not_equal_elem(lhs: BoolTensor<Flex>, rhs: burn_backend::Scalar) -> BoolTensor<Flex> {
        let out_dtype = burn_std::BoolDType::from(lhs.dtype());
        let rhs: bool = rhs.elem();
        crate::ops::comparison::bool_not_equal_elem(lhs, rhs, out_dtype)
    }

    fn bool_any(tensor: BoolTensor<Flex>) -> BoolTensor<Flex> {
        let out_dtype = burn_std::BoolDType::from(tensor.dtype());
        crate::ops::comparison::any_bool(tensor, out_dtype)
    }

    fn bool_any_dim(tensor: BoolTensor<Flex>, dim: usize) -> BoolTensor<Flex> {
        let out_dtype = burn_std::BoolDType::from(tensor.dtype());
        crate::ops::comparison::any_bool_dim(tensor, dim, out_dtype)
    }

    fn bool_all(tensor: BoolTensor<Flex>) -> BoolTensor<Flex> {
        let out_dtype = burn_std::BoolDType::from(tensor.dtype());
        crate::ops::comparison::all_bool(tensor, out_dtype)
    }

    fn bool_all_dim(tensor: BoolTensor<Flex>, dim: usize) -> BoolTensor<Flex> {
        let out_dtype = burn_std::BoolDType::from(tensor.dtype());
        crate::ops::comparison::all_bool_dim(tensor, dim, out_dtype)
    }

    fn bool_select(
        tensor: BoolTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
    ) -> BoolTensor<Flex> {
        crate::ops::gather_scatter::select::<u8>(tensor, dim, indices)
    }

    fn bool_select_or(
        tensor: BoolTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
        value: BoolTensor<Flex>,
    ) -> BoolTensor<Flex> {
        let mut result = crate::ops::gather_scatter::select_add::<u8>(tensor, dim, indices, value);
        // Clamp to 0/1: select_add sums u8 values, but bool OR saturates at 1
        let storage: &mut [u8] = result.storage_mut();
        for v in storage.iter_mut() {
            if *v > 1 {
                *v = 1;
            }
        }
        result
    }

    fn bool_transpose(tensor: BoolTensor<Flex>) -> BoolTensor<Flex> {
        let ndims = tensor.layout().num_dims();
        if ndims < 2 {
            return tensor;
        }
        tensor.transpose(ndims - 2, ndims - 1)
    }

    fn bool_repeat_dim(tensor: BoolTensor<Flex>, dim: usize, times: usize) -> BoolTensor<Flex> {
        crate::ops::repeat_dim::repeat_dim(tensor, dim, times)
    }

    async fn bool_argwhere(tensor: BoolTensor<Flex>, out_dtype: IntDType) -> IntTensor<Flex> {
        let tensor = tensor.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let ndims = shape.num_dims();
        let data: &[u8] = tensor.storage();
        let n = shape.num_elements();

        let count = data[..n].iter().filter(|&&v| v != 0).count();
        let mut coords: Vec<isize> = Vec::with_capacity(count * ndims);
        let strides = crate::layout::contiguous_strides_usize(&shape);

        for (flat_idx, &val) in data[..n].iter().enumerate() {
            if val != 0 {
                let mut remaining = flat_idx;
                for &s in &strides {
                    coords.push((remaining / s) as isize);
                    remaining %= s;
                }
            }
        }

        let out_shape = Shape::from(vec![count, ndims]);
        let result = FlexTensor::new(
            Bytes::from_elems(coords),
            Layout::contiguous(out_shape),
            crate::ops::INDEX_DTYPE,
        );
        if result.dtype() != DType::from(out_dtype) {
            Flex::int_cast(result, out_dtype)
        } else {
            result
        }
    }
}

/// Boolean binary operation type.
#[derive(Clone, Copy)]
enum BoolBinaryOp {
    And,
    Or,
    Xor,
}

fn bool_binary_op_simd(lhs: FlexTensor, rhs: FlexTensor, op: BoolBinaryOp) -> FlexTensor {
    use crate::strided_index::StridedIter;

    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "bool_binary_op: dtype mismatch");

    // Broadcast to a common shape before dispatching. The scalar/SIMD helpers
    // below assume equal-length operands; without this, mismatched shapes
    // either silently keep the lhs shape or OOB-panic inside the helpers.
    let (mut lhs, mut rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    // Preserve the input bool dtype (taken from lhs; rhs is assumed to match
    // in dtype, checked above).
    let out_dtype = burn_std::BoolDType::from(lhs.dtype());
    let shape = lhs.layout().shape().clone();
    let l_offsets = lhs.layout().contiguous_offsets();
    let r_offsets = rhs.layout().contiguous_offsets();

    // Fast path 1: lhs is unique and contiguous at offset 0 -> in-place on lhs
    if lhs.is_unique()
        && let (Some((0, l_end)), Some((r_start, r_end))) = (l_offsets, r_offsets)
    {
        let rhs_storage: &[u8] = rhs.bytes();
        let r_slice = &rhs_storage[r_start..r_end];
        let lhs_storage: &mut [u8] = lhs.storage_mut();
        let l_slice = &mut lhs_storage[..l_end];

        match op {
            BoolBinaryOp::And => crate::simd::bool_and_inplace_u8(l_slice, r_slice),
            BoolBinaryOp::Or => crate::simd::bool_or_inplace_u8(l_slice, r_slice),
            BoolBinaryOp::Xor => crate::simd::bool_xor_inplace_u8(l_slice, r_slice),
        }
        return lhs;
    }

    // Fast path 2: rhs is unique and contiguous at offset 0 -> in-place on rhs
    // (And/Or/Xor are commutative, so we can swap operands)
    if rhs.is_unique()
        && let (Some((l_start, l_end)), Some((0, r_end))) = (l_offsets, r_offsets)
    {
        let lhs_storage: &[u8] = lhs.bytes();
        let l_slice = &lhs_storage[l_start..l_end];
        let rhs_storage: &mut [u8] = rhs.storage_mut();
        let r_slice = &mut rhs_storage[..r_end];

        match op {
            BoolBinaryOp::And => crate::simd::bool_and_inplace_u8(r_slice, l_slice),
            BoolBinaryOp::Or => crate::simd::bool_or_inplace_u8(r_slice, l_slice),
            BoolBinaryOp::Xor => crate::simd::bool_xor_inplace_u8(r_slice, l_slice),
        }
        return rhs;
    }

    // Allocating path: neither tensor is suitable for in-place
    let lhs_storage: &[u8] = lhs.bytes();
    let rhs_storage: &[u8] = rhs.bytes();

    let result: Vec<u8> = match (l_offsets, r_offsets) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            let mut out = vec![0u8; l_slice.len()];
            match op {
                BoolBinaryOp::And => crate::simd::bool_and_u8(l_slice, r_slice, &mut out),
                BoolBinaryOp::Or => crate::simd::bool_or_u8(l_slice, r_slice, &mut out),
                BoolBinaryOp::Xor => crate::simd::bool_xor_u8(l_slice, r_slice, &mut out),
            }
            out
        }
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            match op {
                BoolBinaryOp::And => lhs_iter
                    .zip(rhs_iter)
                    .map(|(li, ri)| lhs_storage[li] & rhs_storage[ri])
                    .collect(),
                BoolBinaryOp::Or => lhs_iter
                    .zip(rhs_iter)
                    .map(|(li, ri)| lhs_storage[li] | rhs_storage[ri])
                    .collect(),
                BoolBinaryOp::Xor => lhs_iter
                    .zip(rhs_iter)
                    .map(|(li, ri)| lhs_storage[li] ^ rhs_storage[ri])
                    .collect(),
            }
        }
    };

    crate::ops::comparison::make_bool_tensor(result, shape, out_dtype)
}

// Tests kept here exercise flex-specific dtype storage selection via
// explicit IntDType/FloatDType. Plain bool ops, bool-to-int/float
// casts, and negative-stride (flipped) bool coverage have been migrated
// to crates/burn-backend-tests/tests/tensor/bool/ops/{logical,cast}.rs
// so they run against every backend. When adding new tests, keep them
// here only if they probe flex dtype dispatch; otherwise add them
// there.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use burn_backend::TensorData;
    use burn_backend::ops::BoolTensorOps;
    use burn_std::{FloatDType, IntDType};

    use crate::{Flex, FlexTensor};

    #[test]
    fn test_bool_into_int_u8() {
        let t = FlexTensor::from_data(TensorData::from([true, false, true]));
        let result = Flex::bool_into_int(t, IntDType::U8);
        assert_eq!(result.dtype(), burn_backend::DType::U8);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1u8, 0, 1]);
    }

    #[test]
    fn test_bool_into_float_f64() {
        let t = FlexTensor::from_data(TensorData::from([true, false, true]));
        let result = Flex::bool_into_float(t, FloatDType::F64);
        assert_eq!(result.dtype(), burn_backend::DType::F64);
        let data: Vec<f64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0f64, 0.0, 1.0]);
    }
}
