//! Binary tensor operations (add, sub, mul, div).

use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::FlexTensor;
use crate::layout::Layout;
use crate::strided_index::StridedIter;

#[cfg(feature = "simd")]
use crate::simd;

/// Operation type for SIMD dispatch.
#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Apply a binary operation element-wise to two tensors.
///
/// Requires tensors to have the same shape. Uses SIMD acceleration for f32
/// when available and both tensors are contiguous.
///
/// Pass `simd_hint` to enable direct SIMD dispatch for standard ops (add/sub/mul/div).
/// Pass `None` for custom operations that have no SIMD fast path.
pub fn binary_op<F32Op, F64Op>(
    lhs: FlexTensor,
    rhs: FlexTensor,
    f32_op: F32Op,
    f64_op: F64Op,
    simd_hint: Option<BinaryOp>,
) -> FlexTensor
where
    F32Op: Fn(f32, f32) -> f32 + Copy,
    F64Op: Fn(f64, f64) -> f64 + Copy,
{
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "binary_op: dtype mismatch");

    // Broadcast tensors to the same shape if needed
    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    let dtype = lhs.dtype();

    match dtype {
        DType::F32 => binary_op_f32(lhs, &rhs, f32_op, simd_hint),
        DType::F64 => binary_op_typed(lhs, &rhs, f64_op),
        DType::F16 => binary_op_typed(lhs, &rhs, |a: f16, b: f16| {
            f16::from_f32(f32_op(a.to_f32(), b.to_f32()))
        }),
        DType::BF16 => binary_op_typed(lhs, &rhs, |a: bf16, b: bf16| {
            bf16::from_f32(f32_op(a.to_f32(), b.to_f32()))
        }),
        _ => panic!("binary_op: unsupported dtype {:?}", dtype),
    }
}

/// Specialized binary operation for f32 with SIMD fast path.
#[cfg(feature = "simd")]
fn binary_op_f32<Op>(
    mut lhs: FlexTensor,
    rhs: &FlexTensor,
    op: Op,
    simd_hint: Option<BinaryOp>,
) -> FlexTensor
where
    Op: Fn(f32, f32) -> f32,
{
    // In-place SIMD fast path: lhs unique, contiguous at offset 0, rhs contiguous
    if let Some(simd_op) = simd_hint
        && lhs.is_unique()
        && let (Some((0, l_end)), Some((r_start, r_end))) = (
            lhs.layout().contiguous_offsets(),
            rhs.layout().contiguous_offsets(),
        )
    {
        let r_slice: &[f32] = &rhs.storage()[r_start..r_end];
        let lhs_storage: &mut [f32] = lhs.storage_mut();
        let l_slice = &mut lhs_storage[..l_end];

        match simd_op {
            BinaryOp::Add => simd::add_inplace_f32(l_slice, r_slice),
            BinaryOp::Sub => simd::sub_inplace_f32(l_slice, r_slice),
            BinaryOp::Mul => simd::mul_inplace_f32(l_slice, r_slice),
            BinaryOp::Div => simd::div_inplace_f32(l_slice, r_slice),
        }
        return lhs;
    }

    // Fallback to generic implementation
    binary_op_typed(lhs, rhs, op)
}

/// Fallback when SIMD is disabled.
#[cfg(not(feature = "simd"))]
fn binary_op_f32<Op>(
    lhs: FlexTensor,
    rhs: &FlexTensor,
    op: Op,
    _simd_hint: Option<BinaryOp>,
) -> FlexTensor
where
    Op: Fn(f32, f32) -> f32,
{
    binary_op_typed(lhs, rhs, op)
}

/// Binary operation with in-place optimization for Pod types.
pub(crate) fn binary_op_typed<E, Op>(mut lhs: FlexTensor, rhs: &FlexTensor, op: Op) -> FlexTensor
where
    E: Element + bytemuck::Pod,
    Op: Fn(E, E) -> E,
{
    let rhs_storage: &[E] = rhs.storage();

    // In-place fast path: lhs unique, contiguous at offset 0, rhs contiguous
    if lhs.is_unique()
        && let (Some((0, l_end)), Some((r_start, r_end))) = (
            lhs.layout().contiguous_offsets(),
            rhs.layout().contiguous_offsets(),
        )
    {
        let lhs_storage: &mut [E] = lhs.storage_mut();
        let r_slice = &rhs_storage[r_start..r_end];
        for (l, &r) in lhs_storage[..l_end].iter_mut().zip(r_slice) {
            *l = op(*l, r);
        }
        return lhs;
    }

    // Allocating path
    let shape = lhs.layout().shape().clone();
    let dtype = lhs.dtype();
    let lhs_storage: &[E] = lhs.storage();

    let result: Vec<E> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        // Both contiguous (but lhs not at offset 0)
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| op(a, b))
                .collect()
        }
        // Fast path for 2D non-contiguous (common for transpose)
        _ if lhs.layout().num_dims() == 2 => {
            apply_2d_strided(lhs_storage, rhs_storage, lhs.layout(), rhs.layout(), op)
        }
        // General fallback
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| op(lhs_storage[li], rhs_storage[ri]))
                .collect()
        }
    };

    make_tensor(result, shape, dtype)
}

/// Fast 2D strided binary operation using row-based iteration.
#[inline]
pub(crate) fn apply_2d_strided<E, R, Op>(
    lhs: &[E],
    rhs: &[E],
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> Vec<R>
where
    E: Copy,
    Op: Fn(E, E) -> R,
{
    let (rows, cols, l_row_stride, l_col_stride) = lhs_layout.as_2d_strides().unwrap();
    let (_, _, r_row_stride, r_col_stride) = rhs_layout.as_2d_strides().unwrap();
    let l_offset = lhs_layout.start_offset() as isize;
    let r_offset = rhs_layout.start_offset() as isize;

    let mut result = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let l_row_start = l_offset + row as isize * l_row_stride;
        let r_row_start = r_offset + row as isize * r_row_stride;
        for col in 0..cols {
            let l_idx = (l_row_start + col as isize * l_col_stride) as usize;
            let r_idx = (r_row_start + col as isize * r_col_stride) as usize;
            result.push(op(lhs[l_idx], rhs[r_idx]));
        }
    }

    result
}

/// Apply a scalar operation to each element of a tensor.
///
/// Attempts in-place mutation when tensor is contiguous at offset 0.
pub fn scalar_op<F32Op, F64Op>(
    tensor: FlexTensor,
    scalar: f64,
    f32_op: F32Op,
    f64_op: F64Op,
) -> FlexTensor
where
    F32Op: Fn(f32, f32) -> f32 + Copy,
    F64Op: Fn(f64, f64) -> f64 + Copy,
{
    let dtype = tensor.dtype();

    match dtype {
        DType::F32 => scalar_op_typed(tensor, scalar as f32, f32_op),
        DType::F64 => scalar_op_typed(tensor, scalar, f64_op),
        DType::F16 => {
            let scalar_f16 = f16::from_f32(scalar as f32);
            let s = scalar_f16.to_f32();
            scalar_op_typed(tensor, scalar_f16, |a: f16, _| {
                f16::from_f32(f32_op(a.to_f32(), s))
            })
        }
        DType::BF16 => {
            let scalar_bf16 = bf16::from_f32(scalar as f32);
            let s = scalar_bf16.to_f32();
            scalar_op_typed(tensor, scalar_bf16, |a: bf16, _| {
                bf16::from_f32(f32_op(a.to_f32(), s))
            })
        }
        _ => panic!("scalar_op: unsupported dtype {:?}", dtype),
    }
}

pub(crate) fn scalar_op_typed<E, Op>(mut tensor: FlexTensor, scalar: E, op: Op) -> FlexTensor
where
    E: Element + bytemuck::Pod,
    Op: Fn(E, E) -> E,
{
    // In-place fast path: unique, contiguous at offset 0
    if tensor.is_unique()
        && let Some((0, end)) = tensor.layout().contiguous_offsets()
    {
        let storage: &mut [E] = tensor.storage_mut();
        for x in storage[..end].iter_mut() {
            *x = op(*x, scalar);
        }
        return tensor;
    }

    // Allocating path
    let shape = tensor.layout().shape().clone();
    let dtype = tensor.dtype();
    let storage: &[E] = tensor.storage();

    let result: Vec<E> = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => storage[start..end].iter().map(|&x| op(x, scalar)).collect(),
        None => StridedIter::new(tensor.layout())
            .map(|i| op(storage[i], scalar))
            .collect(),
    };

    make_tensor(result, shape, dtype)
}

/// Helper to construct a tensor from result data.
fn make_tensor<E: bytemuck::Pod + Send + Sync>(
    data: Vec<E>,
    shape: Shape,
    dtype: DType,
) -> FlexTensor {
    let bytes = Bytes::from_elems(data);
    let layout = Layout::contiguous(shape);
    FlexTensor::new(bytes, layout, dtype)
}

/// Apply a binary operation element-wise to two integer tensors.
///
/// Supports all integer dtypes: I64, I32, I16, I8, U64, U32, U16, U8.
pub fn int_binary_op<Op>(lhs: FlexTensor, rhs: FlexTensor, op: Op) -> FlexTensor
where
    Op: Fn(i64, i64) -> i64 + Copy,
{
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "int_binary_op: dtype mismatch");

    // Broadcast tensors to the same shape if needed
    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    let dtype = lhs.dtype();

    match dtype {
        DType::I64 => binary_op_typed(lhs, &rhs, op),
        DType::I32 => binary_op_typed(lhs, &rhs, |a: i32, b: i32| op(a as i64, b as i64) as i32),
        DType::I16 => binary_op_typed(lhs, &rhs, |a: i16, b: i16| op(a as i64, b as i64) as i16),
        DType::I8 => binary_op_typed(lhs, &rhs, |a: i8, b: i8| op(a as i64, b as i64) as i8),
        // u64 values > i64::MAX wrap to negative i64. This is correct for
        // add/sub/mul/bitwise (two's complement). Div/rem are handled at the call site.
        DType::U64 => binary_op_typed(lhs, &rhs, |a: u64, b: u64| op(a as i64, b as i64) as u64),
        DType::U32 => binary_op_typed(lhs, &rhs, |a: u32, b: u32| op(a as i64, b as i64) as u32),
        DType::U16 => binary_op_typed(lhs, &rhs, |a: u16, b: u16| op(a as i64, b as i64) as u16),
        DType::U8 => binary_op_typed(lhs, &rhs, |a: u8, b: u8| op(a as i64, b as i64) as u8),
        _ => panic!("int_binary_op: unsupported dtype {:?}", dtype),
    }
}

/// Apply a scalar operation to each element of an integer tensor.
/// Note: scalar is truncated to target dtype (matches PyTorch).
pub fn int_scalar_op<Op>(tensor: FlexTensor, scalar: i64, op: Op) -> FlexTensor
where
    Op: Fn(i64, i64) -> i64 + Copy,
{
    let dtype = tensor.dtype();

    match dtype {
        DType::I64 => scalar_op_typed(tensor, scalar, op),
        DType::I32 => scalar_op_typed(tensor, scalar as i32, |a: i32, b: i32| {
            op(a as i64, b as i64) as i32
        }),
        DType::I16 => scalar_op_typed(tensor, scalar as i16, |a: i16, b: i16| {
            op(a as i64, b as i64) as i16
        }),
        DType::I8 => scalar_op_typed(tensor, scalar as i8, |a: i8, b: i8| {
            op(a as i64, b as i64) as i8
        }),
        DType::U64 => scalar_op_typed(tensor, scalar as u64, |a: u64, b: u64| {
            op(a as i64, b as i64) as u64
        }),
        DType::U32 => scalar_op_typed(tensor, scalar as u32, |a: u32, b: u32| {
            op(a as i64, b as i64) as u32
        }),
        DType::U16 => scalar_op_typed(tensor, scalar as u16, |a: u16, b: u16| {
            op(a as i64, b as i64) as u16
        }),
        DType::U8 => scalar_op_typed(tensor, scalar as u8, |a: u8, b: u8| {
            op(a as i64, b as i64) as u8
        }),
        _ => panic!("int_scalar_op: unsupported dtype {:?}", dtype),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use burn_backend::{TensorData, Tolerance};

    // ===================
    // Binary ops: f32
    // ===================

    #[test]
    fn test_binary_add_contiguous_f32() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let data = result.into_data();

        let expected: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_sub_contiguous_f32() {
        let a = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x - y, |x, y| x - y, None);
        let data = result.into_data();

        let expected: Vec<f32> = vec![9.0, 18.0, 27.0, 36.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_mul_contiguous_f32() {
        let a = FlexTensor::from_data(TensorData::new(vec![2.0f32, 3.0, 4.0, 5.0], vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y, None);
        let data = result.into_data();

        let expected: Vec<f32> = vec![2.0, 6.0, 12.0, 20.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_div_contiguous_f32() {
        let a = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(vec![2.0f32, 4.0, 5.0, 8.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x / y, |x, y| x / y, None);
        let data = result.into_data();

        let expected: Vec<f32> = vec![5.0, 5.0, 6.0, 5.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    // ===================
    // Binary ops: f64
    // ===================

    #[test]
    fn test_binary_add_contiguous_f64() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(vec![5.0f64, 6.0, 7.0, 8.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let data = result.into_data();

        let expected: Vec<f64> = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(data.as_slice::<f64>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_mul_contiguous_f64() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.5f64, 2.5, 3.5, 4.5], vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(vec![2.0f64, 2.0, 2.0, 2.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y, None);
        let data = result.into_data();

        let expected: Vec<f64> = vec![3.0, 5.0, 7.0, 9.0];
        assert_eq!(data.as_slice::<f64>().unwrap(), expected.as_slice());
    }

    // ===================
    // Non-contiguous
    // ===================

    #[test]
    fn test_binary_mul_non_contiguous_transposed() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(vec![2.0f32, 3.0, 4.0, 5.0], vec![2, 2]));

        let a_t = a.transpose(0, 1);
        let b_t = b.transpose(0, 1);

        let result = binary_op(a_t, b_t, |x, y| x * y, |x, y| x * y, None);
        let data = result.into_data();

        // a_t = [[1, 3], [2, 4]], b_t = [[2, 4], [3, 5]]
        // result = [[2, 12], [6, 20]]
        let expected: Vec<f32> = vec![2.0, 12.0, 6.0, 20.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_add_non_contiguous_narrowed() {
        // Create [4, 4] tensor and narrow to [2, 4]
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let a = FlexTensor::from_data(TensorData::new(data.clone(), vec![4, 4]));
        let b = FlexTensor::from_data(TensorData::new(data, vec![4, 4]));

        let a_narrow = a.narrow(0, 1, 2); // rows 1-2
        let b_narrow = b.narrow(0, 1, 2);

        let result = binary_op(a_narrow, b_narrow, |x, y| x + y, |x, y| x + y, None);
        let data = result.into_data();

        // rows 1-2: [4,5,6,7], [8,9,10,11] doubled
        let expected: Vec<f32> = vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_mixed_contiguous_non_contiguous() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]));

        // a is contiguous, b is transposed (non-contiguous)
        let b_t = b.transpose(0, 1);

        let result = binary_op(a, b_t, |x, y| x + y, |x, y| x + y, None);
        let data = result.into_data();

        // a = [[1,2], [3,4]], b_t = [[10,30], [20,40]]
        // result = [[11,32], [23,44]]
        let expected: Vec<f32> = vec![11.0, 32.0, 23.0, 44.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    // ===================
    // Scalar ops
    // ===================

    #[test]
    fn test_scalar_add_f32() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], vec![3]));
        let result = scalar_op(a, 10.0, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![11.0, 12.0, 13.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_sub_f32() {
        let a = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0], vec![3]));
        let result = scalar_op(a, 5.0, |x, y| x - y, |x, y| x - y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![5.0, 15.0, 25.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_mul_f32() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));
        let result = scalar_op(a, 3.0, |x, y| x * y, |x, y| x * y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![3.0, 6.0, 9.0, 12.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_div_f32() {
        let a = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]));
        let result = scalar_op(a, 10.0, |x, y| x / y, |x, y| x / y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_add_f64() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.0f64, 2.0, 3.0], vec![3]));
        let result = scalar_op(a, 100.0, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        let expected: Vec<f64> = vec![101.0, 102.0, 103.0];
        assert_eq!(data.as_slice::<f64>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_non_contiguous() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));
        let a_t = a.transpose(0, 1);

        let result = scalar_op(a_t, 10.0, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        // a_t = [[1, 3], [2, 4]] + 10 = [[11, 13], [12, 14]]
        let expected: Vec<f32> = vec![11.0, 13.0, 12.0, 14.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    // ===================
    // Edge cases
    // ===================

    #[test]
    fn test_binary_single_element() {
        let a = FlexTensor::from_data(TensorData::new(vec![5.0f32], vec![1]));
        let b = FlexTensor::from_data(TensorData::new(vec![3.0f32], vec![1]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let data = result.into_data();

        assert_eq!(data.as_slice::<f32>().unwrap(), &[8.0f32]);
    }

    #[test]
    fn test_binary_1d_tensor() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]));
        let b = FlexTensor::from_data(TensorData::new(vec![5.0f32, 4.0, 3.0, 2.0, 1.0], vec![5]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let data = result.into_data();

        let expected: Vec<f32> = vec![6.0, 6.0, 6.0, 6.0, 6.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_3d_tensor() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.0f32; 24], vec![2, 3, 4]));
        let b = FlexTensor::from_data(TensorData::new(vec![2.0f32; 24], vec![2, 3, 4]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y, None);
        let data = result.into_data();

        let expected: Vec<f32> = vec![2.0; 24];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_single_element() {
        let a = FlexTensor::from_data(TensorData::new(vec![7.0f32], vec![1]));
        let result = scalar_op(a, 3.0, |x, y| x * y, |x, y| x * y);
        let data = result.into_data();

        assert_eq!(data.as_slice::<f32>().unwrap(), &[21.0f32]);
    }

    #[test]
    fn test_binary_negative_values() {
        let a = FlexTensor::from_data(TensorData::new(vec![-1.0f32, -2.0, 3.0, 4.0], vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, -3.0, -4.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let data = result.into_data();

        let expected: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_negative_value() {
        let a = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], vec![3]));
        let result = scalar_op(a, -1.0, |x, y| x * y, |x, y| x * y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![-1.0, -2.0, -3.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    // ===================
    // F16 tests
    // ===================

    #[test]
    fn test_binary_add_f16() {
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_vals: Vec<f16> = vec![5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<f16> = vec![6.0, 8.0, 10.0, 12.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(f16::from_f32(0.01)),
        );
    }

    #[test]
    fn test_binary_mul_f16() {
        let a_vals: Vec<f16> = vec![2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_vals: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y, None);
        let expected: Vec<f16> = vec![2.0, 6.0, 12.0, 20.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(f16::from_f32(0.01)),
        );
    }

    #[test]
    fn test_binary_f16_transposed() {
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_vals: Vec<f16> = vec![10.0, 20.0, 30.0, 40.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2])).transpose(0, 1);
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2])).transpose(0, 1);

        // a_t = [[1,3], [2,4]], b_t = [[10,30], [20,40]]
        // result = [[11,33], [22,44]]
        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<f16> = vec![11.0, 33.0, 22.0, 44.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(f16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_scalar_f16() {
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0].into_iter().map(f16::from_f32).collect();
        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![3]));

        let result = scalar_op(a, 10.0, |x, y| x + y, |x, y| x + y);
        let expected: Vec<f16> = vec![11.0, 12.0, 13.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(f16::from_f32(0.01)),
        );
    }

    // ===================
    // BF16 tests
    // ===================

    #[test]
    fn test_binary_add_bf16() {
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let b_vals: Vec<bf16> = vec![5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<bf16> = vec![6.0, 8.0, 10.0, 12.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(bf16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_binary_mul_bf16() {
        let a_vals: Vec<bf16> = vec![2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let b_vals: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y, None);
        let expected: Vec<bf16> = vec![2.0, 6.0, 12.0, 20.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(bf16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_binary_bf16_transposed() {
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let b_vals: Vec<bf16> = vec![10.0, 20.0, 30.0, 40.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2])).transpose(0, 1);
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2])).transpose(0, 1);

        // a_t = [[1,3], [2,4]], b_t = [[10,30], [20,40]]
        // result = [[11,33], [22,44]]
        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<bf16> = vec![11.0, 33.0, 22.0, 44.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(bf16::from_f32(0.5)),
        );
    }

    #[test]
    fn test_scalar_bf16() {
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![3]));

        let result = scalar_op(a, 10.0, |x, y| x + y, |x, y| x + y);
        let expected: Vec<bf16> = vec![11.0, 12.0, 13.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(bf16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_scalar_f16_non_representable() {
        // 0.1 is not exactly representable in f16; verify the scalar is rounded
        // to f16 precision before the op (matching dtype semantics).
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0].into_iter().map(f16::from_f32).collect();
        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![3]));

        let s_f16 = f16::from_f32(0.1);
        let result = scalar_op(a, 0.1, |x, y| x * y, |x, y| x * y);
        let expected: Vec<f16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(|v| f16::from_f32(f16::from_f32(v).to_f32() * s_f16.to_f32()))
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(f16::from_f32(0.001)),
        );
    }

    #[test]
    fn test_scalar_bf16_non_representable() {
        // 1.1 is not exactly representable in bf16; verify dtype rounding.
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![3]));

        let s_bf16 = bf16::from_f32(1.1);
        let result = scalar_op(a, 1.1, |x, y| x * y, |x, y| x * y);
        let expected: Vec<bf16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(|v| bf16::from_f32(bf16::from_f32(v).to_f32() * s_bf16.to_f32()))
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(bf16::from_f32(0.01)),
        );
    }
}
