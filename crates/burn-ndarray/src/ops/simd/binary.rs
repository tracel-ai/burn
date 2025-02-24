use core::{cmp::Ordering, marker::PhantomData, slice};

use macerator::{SimdExt, Vectorizable};
use ndarray::ArrayD;
use pulp::{Arch, Simd};

use crate::{NdArrayElement, NdArrayTensor};

use super::{binary_elemwise::SimdBinop, should_use_simd};

#[allow(clippy::result_large_err)]
pub fn try_binary_simd<
    E: NdArrayElement,
    EOut: NdArrayElement,
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: SimdBinop<T, Out>,
>(
    lhs: NdArrayTensor<E>,
    rhs: NdArrayTensor<E>,
) -> Result<NdArrayTensor<EOut>, (NdArrayTensor<E>, NdArrayTensor<E>)> {
    let lhs_len = lhs.array.len();
    let rhs_len = rhs.array.len();
    if !should_use_simd(lhs_len.max(rhs_len))
        || !lhs.is_contiguous()
        || !rhs.is_contiguous()
        || !can_broadcast(&lhs, &rhs)
    {
        return Err((lhs, rhs));
    }
    // Used to assert traits based on the dynamic `DType`.
    let lhs = unsafe { core::mem::transmute::<NdArrayTensor<E>, NdArrayTensor<T>>(lhs) };
    let rhs = unsafe { core::mem::transmute::<NdArrayTensor<E>, NdArrayTensor<T>>(rhs) };
    let out = match lhs_len.cmp(&rhs_len) {
        Ordering::Less => binary_simd_broadcast_lhs::<T, Out, Op>(lhs, rhs),
        Ordering::Equal => binary_simd_same::<T, Out, Op>(lhs, rhs),
        Ordering::Greater => binary_simd_broadcast_rhs::<T, Out, Op>(lhs, rhs),
    };
    // Used to assert traits based on the dynamic `DType`.
    let out = unsafe { core::mem::transmute::<NdArrayTensor<Out>, NdArrayTensor<EOut>>(out) };
    Ok(out)
}

fn can_broadcast<T: NdArrayElement>(lhs: &NdArrayTensor<T>, rhs: &NdArrayTensor<T>) -> bool {
    if lhs.array.ndim() != rhs.array.ndim() {
        return false;
    }
    for dim in 0..lhs.array.ndim() - 1 {
        if lhs.array.dim()[dim] != rhs.array.dim()[dim] {
            return false;
        }
    }
    true
}

fn binary_simd_same<
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: SimdBinop<T, Out>,
>(
    lhs: NdArrayTensor<T>,
    rhs: NdArrayTensor<T>,
) -> NdArrayTensor<Out> {
    let out = if lhs.array.is_unique() {
        let mut buf = lhs.array.into_owned();
        let lhs = buf.as_slice_mut().unwrap();
        let rhs = rhs.array.as_slice().unwrap();
        let out =
            unsafe { core::mem::transmute::<&mut [T], &mut [Out]>(unsafe_alias_slice_mut(lhs)) };
        binary(lhs, rhs, out, PhantomData::<Op>);
        unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<Out>>(buf) }
    } else if rhs.array.is_unique() {
        let mut buf = rhs.array.into_owned();
        let lhs = lhs.array.as_slice().unwrap();
        let rhs = buf.as_slice_mut().unwrap();
        let out =
            unsafe { core::mem::transmute::<&mut [T], &mut [Out]>(unsafe_alias_slice_mut(rhs)) };
        binary(lhs, rhs, out, PhantomData::<Op>);
        unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<Out>>(buf) }
    } else {
        let mut out = unsafe { ArrayD::uninit(lhs.array.shape()).assume_init() };
        let lhs = lhs.array.as_slice().unwrap();
        let rhs = rhs.array.as_slice().unwrap();
        let out_slice = out.as_slice_mut().unwrap();
        binary(lhs, rhs, out_slice, PhantomData::<Op>);
        out
    };
    NdArrayTensor::new(out.into_shared())
}

fn binary_simd_broadcast_lhs<
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: SimdBinop<T, Out>,
>(
    lhs: NdArrayTensor<T>,
    rhs: NdArrayTensor<T>,
) -> NdArrayTensor<Out> {
    let out = if rhs.array.is_unique() {
        let mut buf = rhs.array.into_owned();
        let lhs = lhs.array.as_slice().unwrap();
        let rhs = buf.as_slice_mut().unwrap();
        let out =
            unsafe { core::mem::transmute::<&mut [T], &mut [Out]>(unsafe_alias_slice_mut(rhs)) };
        for offs in (0..rhs.len()).step_by(lhs.len()) {
            let end = offs + lhs.len();
            binary(lhs, &rhs[offs..end], &mut out[offs..end], PhantomData::<Op>);
        }
        unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<Out>>(buf) }
    } else {
        let mut out = unsafe { ArrayD::uninit(lhs.array.shape()).assume_init() };
        let lhs = lhs.array.as_slice().unwrap();
        let rhs = rhs.array.as_slice().unwrap();
        let dst = out.as_slice_mut().unwrap();
        for offs in (0..rhs.len()).step_by(lhs.len()) {
            let end = offs + lhs.len();
            binary(lhs, &rhs[offs..end], &mut dst[offs..end], PhantomData::<Op>);
        }
        out
    };
    NdArrayTensor::new(out.into_shared())
}

fn binary_simd_broadcast_rhs<
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: SimdBinop<T, Out>,
>(
    lhs: NdArrayTensor<T>,
    rhs: NdArrayTensor<T>,
) -> NdArrayTensor<Out> {
    let out = if lhs.array.is_unique() {
        let mut buf = lhs.array.into_owned();
        let lhs = buf.as_slice_mut().unwrap();
        let rhs = rhs.array.as_slice().unwrap();
        let out =
            unsafe { core::mem::transmute::<&mut [T], &mut [Out]>(unsafe_alias_slice_mut(lhs)) };
        for offs in (0..rhs.len()).step_by(lhs.len()) {
            let end = offs + lhs.len();
            binary(&lhs[offs..end], rhs, &mut out[offs..end], PhantomData::<Op>);
        }
        unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<Out>>(buf) }
    } else {
        let mut out = unsafe { ArrayD::uninit(lhs.array.shape()).assume_init() };
        let lhs = lhs.array.as_slice().unwrap();
        let rhs = rhs.array.as_slice().unwrap();
        let dst = out.as_slice_mut().unwrap();
        for offs in (0..rhs.len()).step_by(lhs.len()) {
            let end = offs + lhs.len();
            binary(&lhs[offs..end], rhs, &mut dst[offs..end], PhantomData::<Op>);
        }
        out
    };
    NdArrayTensor::new(out.into_shared())
}

#[pulp::with_simd(binary = Arch::new())]
fn binary_simd_slice<
    S: Simd,
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: SimdBinop<T, Out>,
>(
    simd: S,
    lhs: &[T],
    rhs: &[T],
    out: &mut [Out],
    _op: PhantomData<Op>,
) {
    let (head_lhs, main_lhs, tail_lhs) = unsafe { lhs.align_to::<T::Vector<S>>() };
    let (head_rhs, main_rhs, tail_rhs) = unsafe { rhs.align_to::<T::Vector<S>>() };
    let (head_out, main_out, tail_out) = unsafe { out.align_to_mut::<Out::Vector<S>>() };
    for ((lhs, rhs), out) in head_lhs.iter().zip(head_rhs).zip(head_out) {
        *out = Op::apply(*lhs, *rhs);
    }
    for ((lhs, rhs), out) in tail_lhs.iter().zip(tail_rhs).zip(tail_out) {
        *out = Op::apply(*lhs, *rhs);
    }
    let mut chunks_lhs = main_lhs.chunks_exact(4);
    let mut chunks_rhs = main_rhs.chunks_exact(4);
    let mut chunks_out = main_out.chunks_exact_mut(4);
    while let Some(((lhs, rhs), out)) = chunks_lhs
        .next()
        .zip(chunks_rhs.next())
        .zip(chunks_out.next())
    {
        unsafe {
            let lhs_ptr = lhs as *const _ as *const T;
            let rhs_ptr = rhs as *const _ as *const T;
            let out_ptr = out as *mut _ as *mut Out;

            let lhs0 = simd.vload(lhs_ptr);
            let lhs1 = simd.vload(lhs_ptr.add(T::lanes::<S>()));
            let lhs2 = simd.vload(lhs_ptr.add(2 * T::lanes::<S>()));
            let lhs3 = simd.vload(lhs_ptr.add(3 * T::lanes::<S>()));

            let rhs0 = simd.vload(rhs_ptr);
            let rhs1 = simd.vload(rhs_ptr.add(T::lanes::<S>()));
            let rhs2 = simd.vload(rhs_ptr.add(2 * T::lanes::<S>()));
            let rhs3 = simd.vload(rhs_ptr.add(3 * T::lanes::<S>()));

            let s0 = Op::apply_vec(simd, lhs0, rhs0);
            let s1 = Op::apply_vec(simd, lhs1, rhs1);
            let s2 = Op::apply_vec(simd, lhs2, rhs2);
            let s3 = Op::apply_vec(simd, lhs3, rhs3);

            simd.vstore(out_ptr, s0);
            simd.vstore(out_ptr.add(T::lanes::<S>()), s1);
            simd.vstore(out_ptr.add(2 * T::lanes::<S>()), s2);
            simd.vstore(out_ptr.add(3 * T::lanes::<S>()), s3);
        }
    }
    for ((lhs, rhs), out) in chunks_lhs
        .remainder()
        .iter()
        .zip(chunks_rhs)
        .zip(chunks_out.into_remainder())
    {
        unsafe {
            let lhs_ptr = lhs as *const _ as *const T;
            let rhs_ptr = rhs as *const _ as *mut T;
            let out_ptr = out as *mut _ as *mut Out;

            let s0 = Op::apply_vec(simd, simd.vload(lhs_ptr), simd.vload(rhs_ptr));
            simd.vstore(out_ptr, s0);
        }
    }
}

fn unsafe_alias_slice_mut<'a, T>(slice: &mut [T]) -> &'a mut [T] {
    let ptr = slice.as_mut_ptr();
    let len = slice.len();
    unsafe { slice::from_raw_parts_mut(ptr, len) }
}
