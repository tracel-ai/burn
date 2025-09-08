use core::marker::PhantomData;

use bytemuck::cast;
use macerator::{
    Scalar, Simd, VAbs, VBitNot, VRecip, Vector, vload, vload_unaligned, vstore, vstore_unaligned,
};
use ndarray::ArrayD;
use num_traits::Signed;
use seq_macro::seq;

use crate::{NdArrayElement, SharedArray};

use super::should_use_simd;

pub trait SimdUnop<T: Scalar, Out: Scalar> {
    fn apply_vec<S: Simd>(input: Vector<S, T>) -> Vector<S, Out>;
    fn apply(input: T) -> Out;
    fn is_accelerated<S: Simd>() -> bool;
}

pub struct RecipVec;

impl SimdUnop<f32, f32> for RecipVec {
    fn apply_vec<S: Simd>(input: Vector<S, f32>) -> Vector<S, f32> {
        input.recip()
    }

    fn apply(input: f32) -> f32 {
        input.recip()
    }

    fn is_accelerated<S: Simd>() -> bool {
        <f32 as VRecip>::is_accelerated::<S>()
    }
}

pub struct VecAbs;

impl<T: VAbs + Signed> SimdUnop<T, T> for VecAbs {
    fn apply_vec<S: Simd>(input: Vector<S, T>) -> Vector<S, T> {
        input.abs()
    }

    fn apply(input: T) -> T {
        input.abs()
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VAbs>::is_accelerated::<S>()
    }
}

pub struct VecBitNot;

impl<T: VBitNot> SimdUnop<T, T> for VecBitNot {
    fn apply_vec<S: Simd>(input: Vector<S, T>) -> Vector<S, T> {
        !input
    }

    fn apply(input: T) -> T {
        input.not()
    }

    fn is_accelerated<S: Simd>() -> bool {
        <T as VBitNot>::is_accelerated::<S>()
    }
}

#[macerator::with_simd]
fn is_accelerated<S: Simd, T: Scalar, Out: Scalar, Op: SimdUnop<T, Out>>(
    _x: PhantomData<(T, Out, Op)>,
) -> bool {
    Op::is_accelerated::<S>()
}

pub fn try_unary_simd<
    E: NdArrayElement,
    EOut: NdArrayElement,
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: SimdUnop<T, Out>,
>(
    input: SharedArray<E>,
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
        unsafe { unary_scalar_simd_inplace::<T, Out, Op>(input) }
    } else {
        unary_scalar_simd_owned::<T, Out, Op>(input)
    };
    // Used to assert traits based on the dynamic `DType`.
    let out = unsafe { core::mem::transmute::<SharedArray<Out>, SharedArray<EOut>>(out) };
    Ok(out)
}

/// Execute operation in line.
/// SAFETY:
/// Must ensure `size_of::<T> == size_of::<Out>` and `align_of::<T> >= align_of::<Out>`.
unsafe fn unary_scalar_simd_inplace<
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: SimdUnop<T, Out>,
>(
    input: SharedArray<T>,
) -> SharedArray<Out> {
    let mut buffer = input.into_owned();
    let slice = buffer.as_slice_memory_order_mut().unwrap();
    // This is only called when in and out have the same size, so it's safe
    unsafe { unary_slice_inplace::<T, Out, Op>(slice, PhantomData) };
    // Buffer has the same elem size and is filled with the operation output, so this is safe
    let out = unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<Out>>(buffer) };
    out.into_shared()
}

fn unary_scalar_simd_owned<
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: SimdUnop<T, Out>,
>(
    input: SharedArray<T>,
) -> SharedArray<Out> {
    let mut out = unsafe { ArrayD::uninit(input.shape()).assume_init() };
    let input = input.as_slice_memory_order().unwrap();
    let out_slice = out.as_slice_memory_order_mut().unwrap();
    unary_slice::<T, Out, Op>(input, out_slice, PhantomData);
    out.into_shared()
}

#[allow(clippy::erasing_op, clippy::identity_op)]
#[macerator::with_simd]
fn unary_slice<
    'a,
    S: Simd,
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: SimdUnop<T, Out>,
>(
    input: &'a [T],
    out: &'a mut [Out],
    _op: PhantomData<Op>,
) where
    'a: 'a,
{
    let lanes = T::lanes::<S>();
    let mut chunks_input = input.chunks_exact(8 * lanes);
    let mut chunks_out = out.chunks_exact_mut(8 * lanes);
    while let Some((input, out)) = chunks_input.next().zip(chunks_out.next()) {
        seq!(N in 0..8 {
            // Load one full vector from `input`.
            // SAFETY: Guaranteed to be in bounds because `len == 8 * lanes`
            let s~N = unsafe { vload_unaligned(&input[N * lanes]) };
            let s~N = Op::apply_vec::<S>(s~N);
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
        let s0 = Op::apply_vec::<S>(s0);
        // Store one full vector to `out`.
        // SAFETY: Guaranteed to be in bounds because `len == lanes`
        unsafe { vstore_unaligned(out.as_mut_ptr(), s0) };
    }

    for (input, out) in chunks_input
        .remainder()
        .iter()
        .zip(chunks_out.into_remainder())
    {
        *out = Op::apply(*input)
    }
}

/// Execute operation in line.
/// SAFETY:
/// Must ensure `size_of::<T> == size_of::<Out>` and `align_of::<T> >= align_of::<Out>`.
#[macerator::with_simd]
unsafe fn unary_slice_inplace<
    'a,
    S: Simd,
    T: NdArrayElement + Scalar,
    Out: NdArrayElement + Scalar,
    Op: SimdUnop<T, Out>,
>(
    buf: &'a mut [T],
    _op: PhantomData<(Out, Op)>,
) where
    'a: 'a,
{
    let (head, main, tail) = unsafe { buf.align_to_mut::<Vector<S, T>>() };
    for elem in head.iter_mut().chain(tail) {
        *elem = cast(Op::apply(*elem));
    }
    let mut chunks = main.chunks_exact_mut(8);
    for elem in chunks.by_ref() {
        seq!(N in 0..8 {
            // Load a full vector from the aligned portion of the buffer.
            // SAFETY: `align_to_mut` guarantees we're aligned to `T::Vector`'s size, and there is
            // always a full vector in bounds.
            let s~N = unsafe { vload(&elem[N] as *const _ as *const T) };
            let s~N = Op::apply_vec::<S>(s~N);
            // Store a full vector at the same position as the input. Cast is safe because `Out` is
            // size and align compatible
            unsafe { vstore(&mut elem[N] as *mut _ as *mut Out, s~N) };
        });
    }

    for elem in chunks.into_remainder() {
        // Load a full vector from the aligned portion of the buffer.
        // SAFETY: `align_to_mut` guarantees we're aligned to `T::Vector`'s size, and there is
        // always a full vector in bounds.
        let s0 = unsafe { vload(elem as *const _ as *const T) };

        let s0 = Op::apply_vec::<S>(s0);
        // Store a full vector at the same position as the input. Cast is safe because `Out` is
        // size and align compatible
        unsafe { vstore(elem as *mut _ as *mut Out, s0) };
    }
}
