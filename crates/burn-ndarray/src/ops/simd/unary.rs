use core::marker::PhantomData;

use macerator::{SimdExt, VAbs, VBitNot, VRecip, VSqrt, Vectorizable};
use ndarray::ArrayD;
use num_traits::Signed;
use pulp::{cast, Arch, Simd};

use crate::{NdArrayElement, NdArrayTensor};

use super::{load4, load4_unaligned, should_use_simd, store4, store4_unaligned};

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

pub struct SqrtVec;

impl SimdUnop<f32, f32> for SqrtVec {
    fn apply_vec<S: Simd>(
        simd: S,
        input: <f32 as Vectorizable>::Vector<S>,
    ) -> <f32 as Vectorizable>::Vector<S> {
        f32::vsqrt(simd, input)
    }

    fn apply(input: f32) -> f32 {
        input.sqrt()
    }
}

impl SimdUnop<f64, f64> for SqrtVec {
    fn apply_vec<S: Simd>(
        simd: S,
        input: <f64 as Vectorizable>::Vector<S>,
    ) -> <f64 as Vectorizable>::Vector<S> {
        f64::vsqrt(simd, input)
    }

    fn apply(input: f64) -> f64 {
        input.sqrt()
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
    let out = if size_of::<T>() == size_of::<Out>() && input.array.is_unique() {
        unary_scalar_simd_inplace::<T, Out, Op>(input)
    } else {
        unary_scalar_simd_owned::<T, Out, Op>(input)
    };
    // Used to assert traits based on the dynamic `DType`.
    let out = unsafe { core::mem::transmute::<NdArrayTensor<Out>, NdArrayTensor<EOut>>(out) };
    Ok(out)
}

fn unary_scalar_simd_inplace<
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
    let mut chunks_input = input.chunks_exact(4 * lanes);
    let mut chunks_out = out.chunks_exact_mut(4 * lanes);
    while let Some((input, out)) = chunks_input.next().zip(chunks_out.next()) {
        unsafe {
            let (s0, s1, s2, s3) = load4_unaligned(simd, input.as_ptr());

            let s0 = Op::apply_vec(simd, s0);
            let s1 = Op::apply_vec(simd, s1);
            let s2 = Op::apply_vec(simd, s2);
            let s3 = Op::apply_vec(simd, s3);

            store4_unaligned(simd, out.as_mut_ptr(), s0, s1, s2, s3);
        }
    }
    let mut chunks_input = chunks_input.remainder().chunks_exact(lanes);
    let mut chunks_out = chunks_out.into_remainder().chunks_exact_mut(lanes);
    while let Some((input, out)) = chunks_input.next().zip(chunks_out.next()) {
        unsafe {
            let s0 = simd.vload_unaligned(input as *const _ as *const T);
            let s0 = Op::apply_vec(simd, s0);

            simd.vstore_unaligned(out as *mut _ as *mut Out, s0);
        }
    }

    for (input, out) in chunks_input
        .remainder()
        .iter()
        .zip(chunks_out.into_remainder())
    {
        *out = Op::apply(*input)
    }
}

#[pulp::with_simd(unary_slice_inplace = Arch::new())]
fn unary_simd_slice_inplace<
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
    let mut chunks = main.chunks_exact_mut(4);
    for elem in chunks.by_ref() {
        unsafe {
            let (s0, s1, s2, s3) = load4(simd, elem as *const _ as *const T);

            let s0 = Op::apply_vec(simd, s0);
            let s1 = Op::apply_vec(simd, s1);
            let s2 = Op::apply_vec(simd, s2);
            let s3 = Op::apply_vec(simd, s3);

            store4(simd, elem as *mut _ as *mut Out, s0, s1, s2, s3);
        }
    }

    for elem in chunks.into_remainder() {
        let input_ptr = elem as *const _ as *const T;
        let out_ptr = elem as *mut _ as *mut Out;

        let s0 = unsafe { simd.vload(input_ptr) };

        let s0 = Op::apply_vec(simd, s0);

        unsafe { simd.vstore(out_ptr, s0) };
    }
}
