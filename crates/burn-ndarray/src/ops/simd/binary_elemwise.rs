use core::marker::PhantomData;

use bytemuck::cast;
use macerator::{
    Scalar, Simd, VAdd, VBitAnd, VBitOr, VBitXor, VDiv, VMul, VOrd, VSub, Vector, vload,
    vload_unaligned, vstore, vstore_unaligned,
};
use ndarray::ArrayD;
use seq_macro::seq;

use crate::{NdArrayElement, SharedArray, ops::simd::uninit_array_like};

use super::{MinMax, should_use_simd};

pub trait ScalarSimdBinop<T: Scalar, Out: Scalar> {
    type Rhs: Copy;
    type RhsVec<S: Simd>: Copy;
    fn splat<S: Simd>(rhs: Self::Rhs) -> Self::RhsVec<S>;
    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Self::RhsVec<S>) -> Vector<S, Out>;
    fn apply(lhs: T, rhs: Self::Rhs) -> Out;
    fn is_accelerated<S: Simd>() -> bool;
}

pub struct VecAdd;
pub struct VecDiv;
pub struct VecMul;
pub struct VecSub;
pub struct VecMin;
pub struct VecMax;
pub struct VecClamp;
pub struct VecBitAnd;
pub struct VecBitOr;
pub struct VecBitXor;

impl<T: VAdd> ScalarSimdBinop<T, T> for VecAdd {
    type Rhs = T;
    type RhsVec<S: Simd> = Vector<S, T>;

    fn splat<S: Simd>(rhs: Self::Rhs) -> Self::RhsVec<S> {
        rhs.splat()
    }

    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Self::RhsVec<S>) -> Vector<S, T> {
        lhs + rhs
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs + rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VAdd>::is_accelerated::<S>()
    }
}

impl<T: VDiv> ScalarSimdBinop<T, T> for VecDiv {
    type Rhs = T;
    type RhsVec<S: Simd> = Vector<S, T>;

    fn splat<S: Simd>(rhs: Self::Rhs) -> Self::RhsVec<S> {
        rhs.splat()
    }

    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Self::RhsVec<S>) -> Vector<S, T> {
        lhs / rhs
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs / rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VDiv>::is_accelerated::<S>()
    }
}

impl<T: VMul> ScalarSimdBinop<T, T> for VecMul {
    type Rhs = T;
    type RhsVec<S: Simd> = Vector<S, T>;

    fn splat<S: Simd>(rhs: Self::Rhs) -> Self::RhsVec<S> {
        rhs.splat()
    }

    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Self::RhsVec<S>) -> Vector<S, T> {
        lhs * rhs
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs * rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VMul>::is_accelerated::<S>()
    }
}

impl<T: VSub> ScalarSimdBinop<T, T> for VecSub {
    type Rhs = T;
    type RhsVec<S: Simd> = Vector<S, T>;

    fn splat<S: Simd>(rhs: Self::Rhs) -> Self::RhsVec<S> {
        rhs.splat()
    }

    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Self::RhsVec<S>) -> Vector<S, T> {
        lhs - rhs
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs - rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VSub>::is_accelerated::<S>()
    }
}

impl<T: VOrd + MinMax> ScalarSimdBinop<T, T> for VecMin {
    type Rhs = T;
    type RhsVec<S: Simd> = Vector<S, T>;

    fn splat<S: Simd>(rhs: Self::Rhs) -> Self::RhsVec<S> {
        rhs.splat()
    }

    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Self::RhsVec<S>) -> Vector<S, T> {
        lhs.min(rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.min(rhs)
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VOrd>::is_min_max_accelerated::<S>()
    }
}

impl<T: VOrd + MinMax> ScalarSimdBinop<T, T> for VecMax {
    type Rhs = T;
    type RhsVec<S: Simd> = Vector<S, T>;

    fn splat<S: Simd>(rhs: Self::Rhs) -> Self::RhsVec<S> {
        rhs.splat()
    }

    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Self::RhsVec<S>) -> Vector<S, T> {
        lhs.max(rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.max(rhs)
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VOrd>::is_min_max_accelerated::<S>()
    }
}

impl<T: VOrd + MinMax> ScalarSimdBinop<T, T> for VecClamp {
    type Rhs = (T, T);
    type RhsVec<S: Simd> = (Vector<S, T>, Vector<S, T>);

    fn splat<S: Simd>((min, max): Self::Rhs) -> Self::RhsVec<S> {
        (min.splat(), max.splat())
    }

    fn apply_vec<S: Simd>(lhs: Vector<S, T>, (min, max): Self::RhsVec<S>) -> Vector<S, T> {
        lhs.min(max).max(min)
    }

    fn apply(lhs: T, (min, max): Self::Rhs) -> T {
        lhs.min(max).max(min)
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VOrd>::is_min_max_accelerated::<S>()
    }
}

impl<T: VBitAnd> ScalarSimdBinop<T, T> for VecBitAnd {
    type Rhs = T;
    type RhsVec<S: Simd> = Vector<S, T>;

    fn splat<S: Simd>(rhs: Self::Rhs) -> Self::RhsVec<S> {
        rhs.splat()
    }

    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Self::RhsVec<S>) -> Vector<S, T> {
        lhs & rhs
    }

    fn apply(lhs: T, rhs: Self::Rhs) -> T {
        lhs & rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VBitAnd>::is_accelerated::<S>()
    }
}

impl<T: VBitOr> ScalarSimdBinop<T, T> for VecBitOr {
    type Rhs = T;
    type RhsVec<S: Simd> = Vector<S, T>;

    fn splat<S: Simd>(rhs: Self::Rhs) -> Self::RhsVec<S> {
        rhs.splat()
    }

    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Self::RhsVec<S>) -> Vector<S, T> {
        lhs | rhs
    }

    fn apply(lhs: T, rhs: Self::Rhs) -> T {
        lhs | rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VBitOr>::is_accelerated::<S>()
    }
}

impl<T: VBitXor> ScalarSimdBinop<T, T> for VecBitXor {
    type Rhs = T;
    type RhsVec<S: Simd> = Vector<S, T>;

    fn splat<S: Simd>(rhs: Self::Rhs) -> Self::RhsVec<S> {
        rhs.splat()
    }

    fn apply_vec<S: Simd>(lhs: Vector<S, T>, rhs: Self::RhsVec<S>) -> Vector<S, T> {
        lhs ^ rhs
    }

    fn apply(lhs: T, rhs: Self::Rhs) -> T {
        lhs ^ rhs
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VBitXor>::is_accelerated::<S>()
    }
}

#[macerator::with_simd]
fn is_accelerated<S: Simd, T: Scalar, Out: Scalar, Op: ScalarSimdBinop<T, Out>>(
    _x: PhantomData<(T, Out, Op)>,
) -> bool {
    Op::is_accelerated::<S>()
}

pub fn try_binary_scalar_simd<
    E: NdArrayElement,
    EOut: NdArrayElement,
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: ScalarSimdBinop<T, Out>,
>(
    input: SharedArray<E>,
    elem: Op::Rhs,
) -> Result<SharedArray<EOut>, SharedArray<E>> {
    if !should_use_simd(input.len())
        || input.as_slice_memory_order().is_none()
        || !is_accelerated::<T, Out, Op>(PhantomData)
    {
        return Err(input);
    }
    // Used to assert traits based on the dynamic `DType`.
    let input = unsafe { core::mem::transmute::<SharedArray<E>, SharedArray<T>>(input) };
    let out = if size_of::<T>() == size_of::<Out>()
        && align_of::<T>() >= align_of::<Out>()
        && input.is_unique()
    {
        unsafe { binary_scalar_simd_inplace::<T, Out, Op>(input, elem) }
    } else {
        binary_scalar_simd_owned::<T, Out, Op>(input, elem)
    };
    // Used to assert traits based on the dynamic `DType`.
    let out = unsafe { core::mem::transmute::<SharedArray<Out>, SharedArray<EOut>>(out) };
    Ok(out)
}

/// Execute operation in place on an owned tensor
/// SAFETY:
/// Must ensure `size_of::<T> == size_of::<Out>` and `align_of::<T> >= align_of::<Out>`.
unsafe fn binary_scalar_simd_inplace<
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: ScalarSimdBinop<T, Out>,
>(
    input: SharedArray<T>,
    elem: Op::Rhs,
) -> SharedArray<Out> {
    let mut buffer = input.into_owned();
    let slice = buffer.as_slice_memory_order_mut().unwrap();
    unsafe { binary_scalar_slice_inplace::<T, Out, Op>(slice, elem, PhantomData) };
    // Buffer has the same elem size and is filled with the operation output, so this is safe
    let out = unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<Out>>(buffer) };
    out.into_shared()
}

/// Create a new copy of the tensor as the output
fn binary_scalar_simd_owned<
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: ScalarSimdBinop<T, Out>,
>(
    input: SharedArray<T>,
    elem: Op::Rhs,
) -> SharedArray<Out> {
    let mut out = uninit_array_like(&input);
    let input = input.as_slice_memory_order().unwrap();
    let out_slice = out.as_slice_memory_order_mut().unwrap();
    binary_scalar_slice::<T, Out, Op>(input, out_slice, elem, PhantomData);
    out.into_shared()
}

#[inline(always)]
#[allow(clippy::erasing_op, clippy::identity_op)]
#[macerator::with_simd]
fn binary_scalar_slice<
    'a,
    S: Simd,
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: ScalarSimdBinop<T, Out>,
>(
    input: &'a [T],
    out: &'a mut [Out],
    rhs: Op::Rhs,
    _op: PhantomData<Op>,
) where
    'a: 'a,
{
    let lanes = T::lanes::<S>();
    let mut chunks_input = input.chunks_exact(8 * lanes);
    let mut chunks_out = out.chunks_exact_mut(8 * lanes);
    let rhs_vec = Op::splat::<S>(rhs);
    while let Some((input, out)) = chunks_input.next().zip(chunks_out.next()) {
        seq!(N in 0..8 {
            // Load one full vector from `input`.
            // SAFETY: Guaranteed to be in bounds because `len == 8 * lanes`
            let s~N = unsafe { vload_unaligned(&input[N * lanes]) };
            let s~N = Op::apply_vec(s~N, rhs_vec);
            // Store one full vector to `out`.
            // SAFETY: Guaranteed to be in bounds because `len == 8 * lanes`
            unsafe { vstore_unaligned(&mut out[N * lanes], s~N) };
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
        unsafe { vstore_unaligned(out.as_mut_ptr(), s0) };
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
unsafe fn binary_scalar_slice_inplace<
    'a,
    S: Simd,
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: ScalarSimdBinop<T, Out>,
>(
    buf: &'a mut [T],
    rhs: Op::Rhs,
    _op: PhantomData<(Out, Op)>,
) where
    'a: 'a,
{
    let (head, main, tail) = unsafe { buf.align_to_mut::<Vector<S, T>>() };
    for elem in head.iter_mut().chain(tail) {
        *elem = cast(Op::apply(*elem, rhs));
    }
    let mut chunks = main.chunks_exact_mut(8);
    let rhs = Op::splat::<S>(rhs);
    for elem in chunks.by_ref() {
        seq!(N in 0..8 {
            // Load a full vector from the aligned portion of the buffer.
            // SAFETY: `align_to_mut` guarantees we're aligned to `T::Vector`'s size, and there is
            // always a full vector in bounds.
            let s~N = unsafe { vload(&elem[N] as *const _ as *const T) };
            let s~N = Op::apply_vec(s~N, rhs);
            // Store a full vector at the same position as the input. Cast is safe because `Out` is
            // size and align compatible
            unsafe { vstore_unaligned(&mut elem[N] as *mut _ as *mut Out, s~N) };
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
        unsafe { vstore(elem as *mut _ as *mut Out, s0) };
    }
}
