use core::{marker::PhantomData, slice};

use burn_backend::Element;
use macerator::{Mask, Scalar, Simd, VEq, VOrd, Vector, vload_unaligned};
use ndarray::ArrayD;
use seq_macro::seq;

use crate::{NdArrayElement, SharedArray, ops::simd::uninit_array_like};

use super::should_use_simd;

pub trait SimdCmpOp<T: Scalar> {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Mask<S, T>;
    fn apply(lhs: T, rhs: T) -> bool;
    fn is_accelerated<S: Simd>() -> bool;
}

pub struct VecEquals;

impl<T: VEq> SimdCmpOp<T> for VecEquals {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Mask<S, T> {
        lhs.eq(rhs)
    }

    fn apply(lhs: T, rhs: T) -> bool {
        lhs == rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VEq>::is_accelerated::<S>()
    }
}

pub struct VecGreater;

impl<T: VOrd + PartialOrd> SimdCmpOp<T> for VecGreater {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Mask<S, T> {
        lhs.gt(rhs)
    }

    fn apply(lhs: T, rhs: T) -> bool {
        lhs > rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VOrd>::is_cmp_accelerated::<S>()
    }
}

pub struct VecGreaterEq;

impl<T: VOrd + PartialOrd> SimdCmpOp<T> for VecGreaterEq {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Mask<S, T> {
        lhs.ge(rhs)
    }

    fn apply(lhs: T, rhs: T) -> bool {
        lhs >= rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VOrd>::is_cmp_accelerated::<S>()
    }
}

pub struct VecLowerEq;

impl<T: VOrd + PartialOrd> SimdCmpOp<T> for VecLowerEq {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Mask<S, T> {
        lhs.le(rhs)
    }

    fn apply(lhs: T, rhs: T) -> bool {
        lhs <= rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VOrd>::is_cmp_accelerated::<S>()
    }
}

pub struct VecLower;

impl<T: VOrd + PartialOrd> SimdCmpOp<T> for VecLower {
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Mask<S, T> {
        lhs.lt(rhs)
    }

    fn apply(lhs: T, rhs: T) -> bool {
        lhs < rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VOrd>::is_cmp_accelerated::<S>()
    }
}

#[macerator::with_simd]
fn is_accelerated<S: Simd, T: Scalar, Op: SimdCmpOp<T>>(_x: PhantomData<(T, Op)>) -> bool {
    Op::is_accelerated::<S>()
}

#[allow(clippy::result_large_err)]
pub fn try_cmp_simd<E: Element, T: NdArrayElement + Scalar, Op: SimdCmpOp<T>>(
    lhs: SharedArray<E>,
    rhs: SharedArray<E>,
) -> Result<SharedArray<bool>, (SharedArray<E>, SharedArray<E>)> {
    let lhs_len = lhs.len();
    let rhs_len = rhs.len();
    if !should_use_simd(lhs_len.max(rhs_len))
        || !lhs.is_standard_layout()
        || !rhs.is_standard_layout()
        || lhs.shape() != rhs.shape()
        || !is_accelerated::<T, Op>(PhantomData)
    {
        return Err((lhs, rhs));
    }
    // Used to assert traits based on the dynamic `DType`.
    let lhs = unsafe { core::mem::transmute::<SharedArray<E>, SharedArray<T>>(lhs) };
    let rhs = unsafe { core::mem::transmute::<SharedArray<E>, SharedArray<T>>(rhs) };
    let out = cmp_simd_same::<T, Op>(lhs, rhs);

    Ok(out)
}

fn cmp_simd_same<T: NdArrayElement + Scalar, Op: SimdCmpOp<T>>(
    lhs: SharedArray<T>,
    rhs: SharedArray<T>,
) -> SharedArray<bool> {
    let out = if lhs.is_unique() && size_of::<T>() == size_of::<bool>() {
        let mut buf = lhs.into_owned();
        let lhs = buf.as_slice_mut().unwrap();
        let rhs = rhs.as_slice().unwrap();
        let out =
            unsafe { core::mem::transmute::<&mut [T], &mut [bool]>(unsafe_alias_slice_mut(lhs)) };
        cmp(lhs, rhs, out, PhantomData::<Op>);
        unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<bool>>(buf) }
    } else if rhs.is_unique() && size_of::<T>() == size_of::<bool>() {
        let mut buf = rhs.into_owned();
        let lhs = lhs.as_slice().unwrap();
        let rhs = buf.as_slice_mut().unwrap();
        let out =
            unsafe { core::mem::transmute::<&mut [T], &mut [bool]>(unsafe_alias_slice_mut(rhs)) };
        cmp(lhs, rhs, out, PhantomData::<Op>);
        unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<bool>>(buf) }
    } else {
        let mut out = uninit_array_like(&lhs);
        let lhs = lhs.as_slice().unwrap();
        let rhs = rhs.as_slice().unwrap();
        let out_slice = out.as_slice_mut().unwrap();
        cmp(lhs, rhs, out_slice, PhantomData::<Op>);
        out
    };
    out.into_shared()
}

#[allow(clippy::erasing_op, clippy::identity_op)]
#[macerator::with_simd]
fn cmp<'a, S: Simd, T: NdArrayElement + Scalar, Op: SimdCmpOp<T>>(
    lhs: &'a [T],
    rhs: &'a [T],
    out: &'a mut [bool],
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
            unsafe { T::mask_store_as_bool(&mut out[N * lanes], s~N) };
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
        unsafe { T::mask_store_as_bool(out.as_mut_ptr(), s0) };
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

pub use elemwise::try_cmp_scalar_simd;

mod elemwise {
    use bytemuck::cast;
    use macerator::vload;

    use super::*;

    pub fn try_cmp_scalar_simd<E: Element, T: NdArrayElement + Scalar, Op: SimdCmpOp<T>>(
        input: SharedArray<E>,
        elem: T,
    ) -> Result<SharedArray<bool>, SharedArray<E>> {
        if !should_use_simd(input.len())
            || input.as_slice_memory_order().is_none()
            || !is_accelerated::<T, Op>(PhantomData)
        {
            return Err(input);
        }
        // Used to assert traits based on the dynamic `DType`.
        let input = unsafe { core::mem::transmute::<SharedArray<E>, SharedArray<T>>(input) };
        let out = if size_of::<T>() == size_of::<bool>()
            && align_of::<T>() >= align_of::<bool>()
            && input.is_unique()
        {
            unsafe { cmp_scalar_simd_inplace::<T, Op>(input, elem) }
        } else {
            cmp_scalar_simd_owned::<T, Op>(input, elem)
        };
        Ok(out)
    }

    /// Execute operation in place on an owned tensor
    /// SAFETY:
    /// Must ensure `size_of::<T> == size_of::<Out>` and `align_of::<T> >= align_of::<Out>`.
    unsafe fn cmp_scalar_simd_inplace<T: NdArrayElement + Scalar, Op: SimdCmpOp<T>>(
        input: SharedArray<T>,
        elem: T,
    ) -> SharedArray<bool> {
        let mut buffer = input.into_owned();
        let slice = buffer.as_slice_memory_order_mut().unwrap();
        unsafe { cmp_scalar_slice_inplace::<T, Op>(slice, elem, PhantomData) };
        // Buffer has the same elem size and is filled with the operation output, so this is safe
        let out = unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<bool>>(buffer) };
        out.into_shared()
    }

    /// Create a new copy of the tensor as the output
    fn cmp_scalar_simd_owned<T: NdArrayElement + Scalar, Op: SimdCmpOp<T>>(
        input: SharedArray<T>,
        elem: T,
    ) -> SharedArray<bool> {
        let mut out = uninit_array_like(&input);
        let input = input.as_slice_memory_order().unwrap();
        let out_slice = out.as_slice_memory_order_mut().unwrap();
        cmp_scalar_slice::<T, Op>(input, out_slice, elem, PhantomData);
        out.into_shared()
    }

    #[inline(always)]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    #[macerator::with_simd]
    fn cmp_scalar_slice<'a, S: Simd, T: NdArrayElement + Scalar, Op: SimdCmpOp<T>>(
        input: &'a [T],
        out: &'a mut [bool],
        rhs: T,
        _op: PhantomData<Op>,
    ) where
        'a: 'a,
    {
        let lanes = T::lanes::<S>();
        let mut chunks_input = input.chunks_exact(8 * lanes);
        let mut chunks_out = out.chunks_exact_mut(8 * lanes);
        let rhs_vec = rhs.splat::<S>();
        while let Some((input, out)) = chunks_input.next().zip(chunks_out.next()) {
            seq!(N in 0..8 {
                // Load one full vector from `input`.
                // SAFETY: Guaranteed to be in bounds because `len == 8 * lanes`
                let s~N = unsafe { vload_unaligned(&input[N * lanes]) };
                let s~N = Op::apply_vec(s~N, rhs_vec);
                // Store one full vector to `out`.
                // SAFETY: Guaranteed to be in bounds because `len == 8 * lanes`
                unsafe { T::mask_store_as_bool(&mut out[N * lanes], s~N) };
            });
        }
        let mut chunks_input = chunks_input.remainder().chunks_exact(lanes);
        let mut chunks_out = chunks_out.into_remainder().chunks_exact_mut(lanes);
        while let Some((input, out)) = chunks_input.next().zip(chunks_out.next()) {
            // Load one full vector from `input`.
            // SAFETY: Guaranteed to be in bounds because `len == lanes`
            let s0 = unsafe { vload_unaligned(input.as_ptr()) };
            let s0 = Op::apply_vec(s0, rhs_vec);
            // Store one full vector to `out`.
            // SAFETY: Guaranteed to be in bounds because `len == lanes`
            unsafe { T::mask_store_as_bool(out.as_mut_ptr(), s0) };
        }

        for (input, out) in chunks_input
            .remainder()
            .iter()
            .zip(chunks_out.into_remainder())
        {
            *out = Op::apply(*input, rhs)
        }
    }

    /// Execute operation in line.
    /// SAFETY:
    /// Must ensure `size_of::<T> == size_of::<Out>` and `align_of::<T> >= align_of::<Out>`.
    #[inline(always)]
    #[macerator::with_simd]
    unsafe fn cmp_scalar_slice_inplace<'a, S: Simd, T: NdArrayElement + Scalar, Op: SimdCmpOp<T>>(
        buf: &'a mut [T],
        rhs: T,
        _op: PhantomData<Op>,
    ) where
        'a: 'a,
    {
        let (head, main, tail) = unsafe { buf.align_to_mut::<Vector<S, T>>() };
        for elem in head.iter_mut().chain(tail) {
            *elem = cast(Op::apply(*elem, rhs));
        }
        let mut chunks = main.chunks_exact_mut(8);
        let rhs = rhs.splat::<S>();
        for elem in chunks.by_ref() {
            seq!(N in 0..8 {
                // Load a full vector from the aligned portion of the buffer.
                // SAFETY: `align_to_mut` guarantees we're aligned to `T::Vector`'s size, and there is
                // always a full vector in bounds.
                let s~N = unsafe { vload(&elem[N] as *const _ as *const T) };
                let s~N = Op::apply_vec(s~N, rhs);
                // Store a full vector at the same position as the input. Cast is safe because `Out` is
                // size and align compatible
                unsafe { T::mask_store_as_bool(&mut elem[N] as *mut _ as *mut bool, s~N) };
            });
        }

        for elem in chunks.into_remainder() {
            // Load a full vector from the aligned portion of the buffer.
            // SAFETY: `align_to_mut` guarantees we're aligned to `T::Vector`'s size, and there is
            // always a full vector in bounds.
            let s0 = unsafe { vload(elem as *const _ as *const T) };

            let s0 = Op::apply_vec(s0, rhs);
            // Store a full vector at the same position as the input. Cast is safe because `Out` is
            // size and align compatible
            unsafe { T::mask_store_as_bool(elem as *mut _ as *mut bool, s0) };
        }
    }
}
