use core::marker::PhantomData;

use macerator::{SimdExt, VAbs, VBitNot, VRecip, Vectorizable};
use ndarray::ArrayD;
use num_traits::Signed;
use pulp::{cast, Arch, Simd};
use seq_macro::seq;

use crate::{NdArrayElement, NdArrayTensor};

use super::should_use_simd;

pub trait SimdUnop<T: Vectorizable, Out: Vectorizable> {
    fn apply_vec<S: Simd>(simd: S, input: T::Vector<S>) -> Out::Vector<S>;
    fn apply(input: T) -> Out;
}

pub struct RecipVec;

impl SimdUnop<f32, f32> for RecipVec {
    fn apply_vec<S: Simd>(
        simd: S,
        input: <f32 as Vectorizable>::Vector<S>,
    ) -> <f32 as Vectorizable>::Vector<S> {
        f32::vrecip(simd, input)
    }

    fn apply(input: f32) -> f32 {
        input.recip()
    }
}

pub struct VecAbs;

impl<T: VAbs + Signed> SimdUnop<T, T> for VecAbs {
    fn apply_vec<S: Simd>(
        simd: S,
        input: <T as Vectorizable>::Vector<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vabs(simd, input)
    }

    fn apply(input: T) -> T {
        input.abs()
    }
}

pub struct VecBitNot;

impl<T: VBitNot> SimdUnop<T, T> for VecBitNot {
    fn apply_vec<S: Simd>(
        simd: S,
        input: <T as Vectorizable>::Vector<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vbitnot(simd, input)
    }

    fn apply(input: T) -> T {
        input.not()
    }
}

pub fn try_unary_simd<
    E: NdArrayElement,
    EOut: NdArrayElement,
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: SimdUnop<T, Out>,
>(
    input: NdArrayTensor<E>,
) -> Result<NdArrayTensor<EOut>, NdArrayTensor<E>> {
    if !should_use_simd(input.array.len()) || input.array.as_slice_memory_order().is_none() {
        return Err(input);
    }
    // Used to assert traits based on the dynamic `DType`.
    let input = unsafe { core::mem::transmute::<NdArrayTensor<E>, NdArrayTensor<T>>(input) };
    let out = if size_of::<T>() == size_of::<Out>()
        && align_of::<T>() >= align_of::<Out>()
        && input.array.is_unique()
    {
        unsafe { unary_scalar_simd_inplace::<T, Out, Op>(input) }
    } else {
        unary_scalar_simd_owned::<T, Out, Op>(input)
    };
    // Used to assert traits based on the dynamic `DType`.
    let out = unsafe { core::mem::transmute::<NdArrayTensor<Out>, NdArrayTensor<EOut>>(out) };
    Ok(out)
}

/// Execute operation in line.
/// SAFETY:
/// Must ensure `size_of::<T> == size_of::<Out>` and `align_of::<T> >= align_of::<Out>`.
unsafe fn unary_scalar_simd_inplace<
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: SimdUnop<T, Out>,
>(
    input: NdArrayTensor<T>,
) -> NdArrayTensor<Out> {
    let mut buffer = input.array.into_owned();
    let slice = buffer.as_slice_memory_order_mut().unwrap();
    // This is only called when in and out have the same size, so it's safe
    unary_slice_inplace::<T, Out, Op>(slice, PhantomData);
    // Buffer has the same elem size and is filled with the operation output, so this is safe
    let out = unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<Out>>(buffer) };
    NdArrayTensor::new(out.into_shared())
}

fn unary_scalar_simd_owned<
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: SimdUnop<T, Out>,
>(
    input: NdArrayTensor<T>,
) -> NdArrayTensor<Out> {
    let mut out = unsafe { ArrayD::uninit(input.array.shape()).assume_init() };
    let input = input.array.as_slice_memory_order().unwrap();
    let out_slice = out.as_slice_memory_order_mut().unwrap();
    unary_slice::<T, Out, Op>(input, out_slice, PhantomData);
    NdArrayTensor::new(out.into_shared())
}

#[allow(clippy::erasing_op, clippy::identity_op)]
#[pulp::with_simd(unary_slice = Arch::new())]
fn unary_simd_slice<
    S: Simd,
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: SimdUnop<T, Out>,
>(
    simd: S,
    input: &[T],
    out: &mut [Out],
    _op: PhantomData<Op>,
) {
    let lanes = T::lanes::<S>();
    let mut chunks_input = input.chunks_exact(8 * lanes);
    let mut chunks_out = out.chunks_exact_mut(8 * lanes);
    while let Some((input, out)) = chunks_input.next().zip(chunks_out.next()) {
        seq!(N in 0..8 {
            // Load one full vector from `input`.
            // SAFETY: Guaranteed to be in bounds because `len == 8 * lanes`
            let s~N = unsafe { simd.vload_unaligned(&input[N * lanes]) };
            let s~N = Op::apply_vec(simd, s~N);
            // Store one full vector to `out`.
            // SAFETY: Guaranteed to be in bounds because `len == 8 * lanes`
            unsafe { simd.vstore_unaligned(&mut out[N * lanes], s~N) };
        });
    }
    let mut chunks_input = chunks_input.remainder().chunks_exact(lanes);
    let mut chunks_out = chunks_out.into_remainder().chunks_exact_mut(lanes);
    while let Some((input, out)) = chunks_input.next().zip(chunks_out.next()) {
        // Load one full vector from `input`.
        // SAFETY: Guaranteed to be in bounds because `len == lanes`
        let s0 = unsafe { simd.vload_unaligned(input.as_ptr()) };
        let s0 = Op::apply_vec(simd, s0);
        // Store one full vector to `out`.
        // SAFETY: Guaranteed to be in bounds because `len == lanes`
        unsafe { simd.vstore_unaligned(out.as_mut_ptr(), s0) };
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
#[pulp::with_simd(unary_slice_inplace = Arch::new())]
unsafe fn unary_simd_slice_inplace<
    S: Simd,
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: SimdUnop<T, Out>,
>(
    simd: S,
    buf: &mut [T],
    _op: PhantomData<(Out, Op)>,
) {
    let (head, main, tail) = unsafe { buf.align_to_mut::<T::Vector<S>>() };
    for elem in head.iter_mut().chain(tail) {
        *elem = cast(Op::apply(*elem));
    }
    let mut chunks = main.chunks_exact_mut(8);
    for elem in chunks.by_ref() {
        seq!(N in 0..8 {
            // Load a full vector from the aligned portion of the buffer.
            // SAFETY: `align_to_mut` guarantees we're aligned to `T::Vector`'s size, and there is
            // always a full vector in bounds.
            let s~N = unsafe { simd.vload(&elem[N] as *const _ as *const T) };
            let s~N = Op::apply_vec(simd, s~N);
            // Store a full vector at the same position as the input. Cast is safe because `Out` is
            // size and align compatible
            unsafe { simd.vstore(&mut elem[N] as *mut _ as *mut Out, s~N) };
        });
    }

    for elem in chunks.into_remainder() {
        // Load a full vector from the aligned portion of the buffer.
        // SAFETY: `align_to_mut` guarantees we're aligned to `T::Vector`'s size, and there is
        // always a full vector in bounds.
        let s0 = unsafe { simd.vload(elem as *const _ as *const T) };

        let s0 = Op::apply_vec(simd, s0);
        // Store a full vector at the same position as the input. Cast is safe because `Out` is
        // size and align compatible
        unsafe { simd.vstore(elem as *mut _ as *mut Out, s0) };
    }
}
