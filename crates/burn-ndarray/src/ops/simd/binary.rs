use core::{cmp::Ordering, marker::PhantomData, slice};

use burn_tensor::Element;
use macerator::{
    SimdExt, VAdd, VBitAnd, VBitOr, VBitXor, VDiv, VEq, VMul, VOrd, VSub, Vectorizable,
};
use ndarray::ArrayD;
use pulp::{cast, Arch, Simd};

use crate::{NdArrayElement, NdArrayTensor};

use super::{
    binary_elemwise::{
        VecAdd, VecBitAnd, VecBitOr, VecBitXor, VecDiv, VecEq, VecMax, VecMin, VecMul, VecSub,
    },
    load4_unaligned, should_use_simd, store4_unaligned, MinMax,
};

pub trait SimdBinop<T: Vectorizable, Out: Vectorizable> {
    fn apply_vec<S: Simd>(simd: S, lhs: T::Vector<S>, rhs: T::Vector<S>) -> Out::Vector<S>;
    fn apply(lhs: T, rhs: T) -> Out;
}

impl<T: VAdd> SimdBinop<T, T> for VecAdd {
    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: <T as Vectorizable>::Vector<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vadd(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.add(rhs)
    }
}

impl<T: VDiv> SimdBinop<T, T> for VecDiv {
    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: <T as Vectorizable>::Vector<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vdiv(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.div(rhs)
    }
}

impl<T: VMul> SimdBinop<T, T> for VecMul {
    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: <T as Vectorizable>::Vector<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vmul(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.mul(rhs)
    }
}

impl<T: VSub> SimdBinop<T, T> for VecSub {
    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: <T as Vectorizable>::Vector<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vsub(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.sub(rhs)
    }
}

impl<T: VOrd + MinMax> SimdBinop<T, T> for VecMin {
    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: <T as Vectorizable>::Vector<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vmin(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        MinMax::min(lhs, rhs)
    }
}

impl<T: VOrd + MinMax> SimdBinop<T, T> for VecMax {
    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: <T as Vectorizable>::Vector<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vmax(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        MinMax::max(lhs, rhs)
    }
}

impl<T: VBitAnd> SimdBinop<T, T> for VecBitAnd {
    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: <T as Vectorizable>::Vector<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vbitand(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.bitand(rhs)
    }
}

impl<T: VBitOr> SimdBinop<T, T> for VecBitOr {
    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: <T as Vectorizable>::Vector<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vbitor(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.bitor(rhs)
    }
}

impl<T: VBitXor> SimdBinop<T, T> for VecBitXor {
    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: <T as Vectorizable>::Vector<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vbitxor(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.bitxor(rhs)
    }
}

impl SimdBinop<u8, u8> for VecEq {
    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <u8 as Vectorizable>::Vector<S>,
        rhs: <u8 as Vectorizable>::Vector<S>,
    ) -> <u8 as Vectorizable>::Vector<S> {
        // Need to bitand the mask with `0b1`, since Rust uses `0` and `1` for bool but mask is `0` or `-1`
        let mask = u8::veq(simd, lhs, rhs);
        let true_ = simd.splat(1u8);
        u8::vbitand(simd, cast(mask), true_)
    }

    fn apply(lhs: u8, rhs: u8) -> u8 {
        (lhs == rhs) as u8
    }
}

#[allow(clippy::result_large_err)]
pub fn try_binary_simd<
    E: Element,
    EOut: Element,
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
        || !lhs.array.is_standard_layout()
        || !rhs.array.is_standard_layout()
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

fn can_broadcast<T: Element>(lhs: &NdArrayTensor<T>, rhs: &NdArrayTensor<T>) -> bool {
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
    println!(
        "lhs unique: {}, rhs unique: {}",
        lhs.array.is_unique(),
        rhs.array.is_unique()
    );
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
        println!("lhs: {lhs:?}");
        println!("rhs: {rhs:?}");
        binary(lhs, rhs, out_slice, PhantomData::<Op>);
        println!("out: {out_slice:?}");
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
    let lanes = T::lanes::<S>();
    let mut chunks_lhs = lhs.chunks_exact(4 * lanes);
    let mut chunks_rhs = rhs.chunks_exact(4 * lanes);
    let mut chunks_out = out.chunks_exact_mut(4 * lanes);
    while let Some(((lhs, rhs), out)) = chunks_lhs
        .next()
        .zip(chunks_rhs.next())
        .zip(chunks_out.next())
    {
        unsafe {
            let (lhs0, lhs1, lhs2, lhs3) = load4_unaligned(simd, lhs as *const _ as *const T);
            let (rhs0, rhs1, rhs2, rhs3) = load4_unaligned(simd, rhs as *const _ as *const T);

            let s0 = Op::apply_vec(simd, lhs0, rhs0);
            let s1 = Op::apply_vec(simd, lhs1, rhs1);
            let s2 = Op::apply_vec(simd, lhs2, rhs2);
            let s3 = Op::apply_vec(simd, lhs3, rhs3);

            store4_unaligned(simd, out as *mut _ as *mut Out, s0, s1, s2, s3);
        }
    }
    let mut chunks_lhs = chunks_lhs.remainder().chunks_exact(lanes);
    let mut chunks_rhs = chunks_rhs.remainder().chunks_exact(lanes);
    let mut chunks_out = chunks_out.into_remainder().chunks_exact_mut(lanes);
    while let Some(((lhs, rhs), out)) = chunks_lhs
        .next()
        .zip(chunks_rhs.next())
        .zip(chunks_out.next())
    {
        unsafe {
            let lhs0 = simd.vload_unaligned(lhs as *const _ as *const T);
            let rhs0 = simd.vload_unaligned(rhs as *const _ as *const T);

            let s0 = Op::apply_vec(simd, lhs0, rhs0);

            simd.vstore_unaligned(out as *mut _ as *mut Out, s0);
        }
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

fn unsafe_alias_slice_mut<'a, T>(slice: &mut [T]) -> &'a mut [T] {
    let ptr = slice.as_mut_ptr();
    let len = slice.len();
    unsafe { slice::from_raw_parts_mut(ptr, len) }
}
