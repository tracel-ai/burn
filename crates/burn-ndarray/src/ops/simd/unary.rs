use core::{marker::PhantomData, slice};

use macerator::{SimdExt, VAbs, VBitNot, VRecip, VSqrt, Vectorizable};
use ndarray::ArrayD;
use num_traits::Signed;
use pulp::{Arch, Simd};

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
    let input = unsafe_alias_slice_mut(slice);
    // This is only called when in and out have the same size, so it's safe
    let out = unsafe { core::mem::transmute::<&mut [T], &mut [Out]>(slice) };
    unary_slice::<T, Out, Op>(input, out, PhantomData);
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
    let (head, main, tail) = unsafe { input.align_to::<T::Vector<S>>() };
    let (out_head, out_main, out_tail) = unsafe { out.align_to_mut::<Out::Vector<S>>() };
    for (a, b) in head.iter().zip(out_head.iter_mut()) {
        *b = Op::apply(*a);
    }
    for (a, b) in tail.iter().zip(out_tail.iter_mut()) {
        *b = Op::apply(*a);
    }
    let mut chunks = main.chunks_exact(4);
    let mut chunks_out = out_main.chunks_exact_mut(4);
    while let Some((a, b)) = chunks.next().zip(chunks_out.next()) {
        unsafe {
            let a_ptr = a as *const _ as *const T;
            let b_ptr = b as *mut _ as *mut Out;

            let s0 = Op::apply_vec(simd, simd.vload(a_ptr));
            let s1 = Op::apply_vec(simd, simd.vload(a_ptr.add(T::lanes::<S>())));
            let s2 = Op::apply_vec(simd, simd.vload(a_ptr.add(2 * T::lanes::<S>())));
            let s3 = Op::apply_vec(simd, simd.vload(a_ptr.add(3 * T::lanes::<S>())));

            simd.vstore(b_ptr, s0);
            simd.vstore(b_ptr.add(T::lanes::<S>()), s1);
            simd.vstore(b_ptr.add(2 * T::lanes::<S>()), s2);
            simd.vstore(b_ptr.add(3 * T::lanes::<S>()), s3);
        }
    }
    for (a, b) in chunks
        .remainder()
        .iter()
        .zip(chunks_out.into_remainder().iter_mut())
    {
        unsafe {
            let a_ptr = a as *const _ as *const T;
            let b_ptr = b as *mut _ as *mut Out;

            let s0 = Op::apply_vec(simd, simd.vload(a_ptr));
            simd.vstore(b_ptr, s0);
        }
    }
}

fn unsafe_alias_slice_mut<'a, T>(slice: &mut [T]) -> &'a mut [T] {
    let ptr = slice.as_mut_ptr();
    let len = slice.len();
    unsafe { slice::from_raw_parts_mut(ptr, len) }
}
