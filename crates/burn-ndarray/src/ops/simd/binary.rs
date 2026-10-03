use core::{marker::PhantomData, slice};

use burn_backend::Element;
use macerator::{
    Scalar, Simd, VAdd, VBitAnd, VBitOr, VBitXor, VDiv, VMul, VOrd, VSub, Vector, vload_unaligned,
    vstore_unaligned,
};
use ndarray::ArrayD;
use seq_macro::seq;

use crate::{NdArrayElement, SharedArray, ops::simd::uninit_array_like};

use super::{
    MinMax,
    binary_elemwise::{
        VecAdd, VecBitAnd, VecBitOr, VecBitXor, VecDiv, VecMax, VecMin, VecMul, VecSub,
    },
    should_use_simd,
};

pub trait SimdBinop<T: Scalar, Out: Scalar> {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, Out>;
    fn apply(lhs: T, rhs: T) -> Out;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<T: VAdd> SimdBinop<T, T> for VecAdd {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T> {
        lhs + rhs
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs + rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VAdd>::is_accelerated::<S>()
    }
}

impl<T: VDiv> SimdBinop<T, T> for VecDiv {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T> {
        lhs / rhs
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs / rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VDiv>::is_accelerated::<S>()
    }
}

impl<T: VMul> SimdBinop<T, T> for VecMul {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T> {
        lhs * rhs
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs * rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VMul>::is_accelerated::<S>()
    }
}

impl<T: VSub> SimdBinop<T, T> for VecSub {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T> {
        lhs - rhs
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs - rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VSub>::is_accelerated::<S>()
    }
}

impl<T: VOrd + MinMax> SimdBinop<T, T> for VecMin {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T> {
        lhs.min(rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        MinMax::min(lhs, rhs)
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VOrd>::is_min_max_accelerated::<S>()
    }
}

impl<T: VOrd + MinMax> SimdBinop<T, T> for VecMax {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T> {
        lhs.max(rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        MinMax::max(lhs, rhs)
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VOrd>::is_min_max_accelerated::<S>()
    }
}

impl<T: VBitAnd> SimdBinop<T, T> for VecBitAnd {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T> {
        lhs & rhs
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.bitand(rhs)
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VBitAnd>::is_accelerated::<S>()
    }
}

impl<T: VBitOr> SimdBinop<T, T> for VecBitOr {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T> {
        lhs | rhs
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.bitor(rhs)
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VBitOr>::is_accelerated::<S>()
    }
}

impl<T: VBitXor> SimdBinop<T, T> for VecBitXor {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T> {
        lhs ^ rhs
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.bitxor(rhs)
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VBitXor>::is_accelerated::<S>()
    }
}

#[macerator::with_simd]
fn is_accelerated<S: Simd, T: Scalar, Out: Scalar, Op: SimdBinop<T, Out>>(
    _x: PhantomData<(T, Out, Op)>,
) -> bool {
    Op::is_accelerated::<S>()
}

#[allow(clippy::result_large_err)]
pub fn try_binary_simd<
    E: Element,
    EOut: Element,
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: SimdBinop<T, Out>,
>(
    lhs: SharedArray<E>,
    rhs: SharedArray<E>,
) -> Result<SharedArray<EOut>, (SharedArray<E>, SharedArray<E>)> {
    let lhs_len = lhs.len();
    let rhs_len = rhs.len();
    if !should_use_simd(lhs_len.max(rhs_len))
        || !lhs.is_standard_layout()
        || !rhs.is_standard_layout()
        || lhs.shape() != rhs.shape()
        || !is_accelerated::<T, Out, Op>(PhantomData)
    {
        return Err((lhs, rhs));
    }
    // Used to assert traits based on the dynamic `DType`.
    let lhs = unsafe { core::mem::transmute::<SharedArray<E>, SharedArray<T>>(lhs) };
    let rhs = unsafe { core::mem::transmute::<SharedArray<E>, SharedArray<T>>(rhs) };
    let out = binary_simd_same::<T, Out, Op>(lhs, rhs);

    // Used to assert traits based on the dynamic `DType`.
    let out = unsafe { core::mem::transmute::<SharedArray<Out>, SharedArray<EOut>>(out) };
    Ok(out)
}

fn binary_simd_same<
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: SimdBinop<T, Out>,
>(
    lhs: SharedArray<T>,
    rhs: SharedArray<T>,
) -> SharedArray<Out> {
    let out = if lhs.is_unique() {
        let mut buf = lhs.into_owned();
        let lhs = buf.as_slice_mut().unwrap();
        let rhs = rhs.as_slice().unwrap();
        let out =
            unsafe { core::mem::transmute::<&mut [T], &mut [Out]>(unsafe_alias_slice_mut(lhs)) };
        binary(lhs, rhs, out, PhantomData::<Op>);
        unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<Out>>(buf) }
    } else if rhs.is_unique() {
        let mut buf = rhs.into_owned();
        let lhs = lhs.as_slice().unwrap();
        let rhs = buf.as_slice_mut().unwrap();
        let out =
            unsafe { core::mem::transmute::<&mut [T], &mut [Out]>(unsafe_alias_slice_mut(rhs)) };
        binary(lhs, rhs, out, PhantomData::<Op>);
        unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<Out>>(buf) }
    } else {
        let mut out = uninit_array_like(&lhs);
        let lhs = lhs.as_slice().unwrap();
        let rhs = rhs.as_slice().unwrap();
        let out_slice = out.as_slice_mut().unwrap();
        binary(lhs, rhs, out_slice, PhantomData::<Op>);
        out
    };
    out.into_shared()
}

#[allow(clippy::erasing_op, clippy::identity_op)]
#[macerator::with_simd]
fn binary<
    'a,
    S: Simd,
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: SimdBinop<T, Out>,
>(
    lhs: &'a [T],
    rhs: &'a [T],
    out: &'a mut [Out],
    _op: PhantomData<Op>,
) where
    'a: 'a,
{
    let lanes = T::lanes::<S>();
    let mut chunks_lhs = lhs.chunks_exact(8 * lanes);
    let mut chunks_rhs = rhs.chunks_exact(8 * lanes);
    let mut chunks_out = out.chunks_exact_mut(8 * lanes);
    while let Some(((lhs, rhs), out)) = chunks_lhs
        .next()
        .zip(chunks_rhs.next())
        .zip(chunks_out.next())
    {
        seq!(N in 0..8 {
            // Load one full vector from `lhs`.
            // SAFETY: Guaranteed to be in bounds because `len == 8 * lanes`
            let lhs~N = unsafe { vload_unaligned::<S, _>(&lhs[N * lanes]) };
            // Load one full vector from `rhs`.
            // SAFETY: Guaranteed to be in bounds because `len == 8 * lanes`
            let rhs~N = unsafe { vload_unaligned(&rhs[N * lanes]) };
            let s~N = Op::apply_vec(lhs~N, rhs~N);
            // Store one full vector to `out`.
            // SAFETY: Guaranteed to be in bounds because `len == 8 * lanes`
            unsafe { vstore_unaligned(&mut out[N * lanes], s~N) };
        });
    }
    let mut chunks_lhs = chunks_lhs.remainder().chunks_exact(lanes);
    let mut chunks_rhs = chunks_rhs.remainder().chunks_exact(lanes);
    let mut chunks_out = chunks_out.into_remainder().chunks_exact_mut(lanes);
    while let Some(((lhs, rhs), out)) = chunks_lhs
        .next()
        .zip(chunks_rhs.next())
        .zip(chunks_out.next())
    {
        // Load one full vector from `lhs`.
        // SAFETY: Guaranteed to be in bounds because `len == lanes`
        let lhs0 = unsafe { vload_unaligned::<S, _>(lhs.as_ptr()) };
        // Load one full vector from `rhs`.
        // SAFETY: Guaranteed to be in bounds because `len == lanes`
        let rhs0 = unsafe { vload_unaligned(rhs.as_ptr()) };
        let s0 = Op::apply_vec(lhs0, rhs0);
        // Store one full vector to `out`.
        // SAFETY: Guaranteed to be in bounds because `len == lanes`
        unsafe { vstore_unaligned(out.as_mut_ptr(), s0) };
    }

    for ((lhs, rhs), out) in chunks_lhs
        .remainder()
        .iter()
        .zip(chunks_rhs.remainder())
        .zip(chunks_out.into_remainder())
    {
        *out = Op::apply(*lhs, *rhs)
    }
}

/// Unsafely alias a slice to use as an inline argument
fn unsafe_alias_slice_mut<'a, T>(slice: &mut [T]) -> &'a mut [T] {
    let ptr = slice.as_mut_ptr();
    let len = slice.len();
    unsafe { slice::from_raw_parts_mut(ptr, len) }
}

#[cfg(test)]
mod tests {
    use core::fmt::Debug;

    use super::*;
    use crate::ops::simd::testutil::{arr, arr_at, f_order, pos, simd, vals, vals_b};

    /// Assert `lhs OP rhs` equals `f` applied elementwise.
    fn binop<T, Out, Op>(lhs: SharedArray<T>, rhs: SharedArray<T>, f: impl Fn(T, T) -> Out)
    where
        T: NdArrayElement + Scalar,
        Out: NdArrayElement + Scalar + PartialEq + Debug,
        Op: SimdBinop<T, Out>,
    {
        let expected = arr(lhs
            .iter()
            .copied()
            .zip(rhs.iter().copied())
            .map(|(a, b)| f(a, b))
            .collect());
        assert_eq!(
            simd(try_binary_simd::<T, Out, T, Out, Op>(lhs, rhs)),
            expected
        );
    }

    #[test]
    fn matches_scalar_for_each_op() {
        binop::<f32, f32, VecDiv>(arr(vals(97)), arr(pos(97)), |a, b| a / b);
        binop::<f32, f32, VecMin>(arr(vals(97)), arr(vals_b(97)), |a, b| a.min(b));
        binop::<f32, f32, VecMax>(arr(vals(97)), arr(vals_b(97)), |a, b| a.max(b));
        binop::<i32, i32, VecBitAnd>(arr(vals(97)), arr(vals_b(97)), |a, b| a & b);
        binop::<i32, i32, VecBitOr>(arr(vals(97)), arr(vals_b(97)), |a, b| a | b);
        binop::<i32, i32, VecBitXor>(arr(vals(97)), arr(vals_b(97)), |a, b| a ^ b);
    }

    #[test]
    fn covers_operand_buffer_reuse() {
        // Unique lhs: the output is written into the lhs buffer.
        binop::<f32, f32, VecAdd>(arr(vals(97)), arr(vals_b(97)), |a, b| a + b);

        // Shared lhs + unique rhs: the output is written into the rhs buffer.
        let lhs = arr(vals(97));
        let _shared = lhs.clone();
        binop::<f32, f32, VecSub>(lhs, arr(vals_b(97)), |a, b| a - b);

        // Both shared: a fresh output buffer is allocated.
        let (lhs, rhs) = (arr(vals(97)), arr(vals_b(97)));
        let _shared = (lhs.clone(), rhs.clone());
        binop::<f32, f32, VecMul>(lhs, rhs, |a, b| a * b);
    }

    #[test]
    fn rejects_mismatched_shapes() {
        assert!(
            try_binary_simd::<f32, f32, f32, f32, VecAdd>(arr(vals(97)), arr(vals(64))).is_err()
        );
    }

    #[test]
    fn rejects_non_standard_layout() {
        let rhs = f_order([12, 8], vals(96));
        assert!(!rhs.is_standard_layout());
        assert!(
            try_binary_simd::<f32, f32, f32, f32, VecAdd>(arr_at(&[12, 8], vals(96)), rhs).is_err()
        );
    }
}
