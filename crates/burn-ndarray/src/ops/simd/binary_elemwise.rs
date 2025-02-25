use core::{marker::PhantomData, slice};

use macerator::{
    SimdExt, VAdd, VBitAnd, VBitOr, VBitXor, VDiv, VEq, VMul, VOrd, VSub, Vectorizable,
};
use ndarray::ArrayD;
use pulp::{cast, Arch, Simd};

use crate::{NdArrayElement, NdArrayTensor};

use super::{should_use_simd, MinMax};

pub trait SimdBinop<T: Vectorizable, Out: Vectorizable> {
    fn apply_vec<S: Simd>(simd: S, lhs: T::Vector<S>, rhs: T::Vector<S>) -> Out::Vector<S>;
    fn apply(lhs: T, rhs: T) -> Out;
}

pub trait ScalarSimdBinop<T: Vectorizable, Out: Vectorizable> {
    type Rhs: Copy;
    type RhsVec<S: Simd>: Copy;
    fn splat<S: Simd>(simd: S, rhs: Self::Rhs) -> Self::RhsVec<S>;
    fn apply_vec<S: Simd>(simd: S, lhs: T::Vector<S>, rhs: Self::RhsVec<S>) -> Out::Vector<S>;
    fn apply(lhs: T, rhs: Self::Rhs) -> Out;
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
pub struct VecEq;

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

impl<T: VAdd> ScalarSimdBinop<T, T> for VecAdd {
    type Rhs = T;
    type RhsVec<S: Simd> = T::Vector<S>;

    fn splat<S: Simd>(simd: S, rhs: Self::Rhs) -> Self::RhsVec<S> {
        simd.splat(rhs)
    }

    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: Self::RhsVec<S>,
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

impl<T: VDiv> ScalarSimdBinop<T, T> for VecDiv {
    type Rhs = T;
    type RhsVec<S: Simd> = T::Vector<S>;

    fn splat<S: Simd>(simd: S, rhs: Self::Rhs) -> Self::RhsVec<S> {
        simd.splat(rhs)
    }

    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: Self::RhsVec<S>,
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

impl<T: VMul> ScalarSimdBinop<T, T> for VecMul {
    type Rhs = T;
    type RhsVec<S: Simd> = T::Vector<S>;

    fn splat<S: Simd>(simd: S, rhs: Self::Rhs) -> Self::RhsVec<S> {
        simd.splat(rhs)
    }

    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: Self::RhsVec<S>,
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

impl<T: VSub> ScalarSimdBinop<T, T> for VecSub {
    type Rhs = T;
    type RhsVec<S: Simd> = T::Vector<S>;

    fn splat<S: Simd>(simd: S, rhs: Self::Rhs) -> Self::RhsVec<S> {
        simd.splat(rhs)
    }

    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: Self::RhsVec<S>,
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

impl<T: VOrd + MinMax> ScalarSimdBinop<T, T> for VecMin {
    type Rhs = T;
    type RhsVec<S: Simd> = T::Vector<S>;

    fn splat<S: Simd>(simd: S, rhs: Self::Rhs) -> Self::RhsVec<S> {
        simd.splat(rhs)
    }

    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: Self::RhsVec<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vmin(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.min(rhs)
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

impl<T: VOrd + MinMax> ScalarSimdBinop<T, T> for VecMax {
    type Rhs = T;
    type RhsVec<S: Simd> = T::Vector<S>;

    fn splat<S: Simd>(simd: S, rhs: Self::Rhs) -> Self::RhsVec<S> {
        simd.splat(rhs)
    }

    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: Self::RhsVec<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vmax(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: T) -> T {
        lhs.max(rhs)
    }
}

impl<T: VOrd + MinMax> ScalarSimdBinop<T, T> for VecClamp {
    type Rhs = (T, T);
    type RhsVec<S: Simd> = (T::Vector<S>, T::Vector<S>);

    fn splat<S: Simd>(simd: S, (min, max): Self::Rhs) -> Self::RhsVec<S> {
        (simd.splat(min), simd.splat(max))
    }

    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        (min, max): Self::RhsVec<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        let s0 = T::vmin(simd, lhs, max);
        T::vmax(simd, s0, min)
    }

    fn apply(lhs: T, (min, max): Self::Rhs) -> T {
        lhs.min(max).max(min)
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

impl<T: VBitAnd> ScalarSimdBinop<T, T> for VecBitAnd {
    type Rhs = T;
    type RhsVec<S: Simd> = T::Vector<S>;

    fn splat<S: Simd>(simd: S, rhs: Self::Rhs) -> Self::RhsVec<S> {
        simd.splat(rhs)
    }

    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: Self::RhsVec<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vbitand(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: Self::Rhs) -> T {
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

impl<T: VBitOr> ScalarSimdBinop<T, T> for VecBitOr {
    type Rhs = T;
    type RhsVec<S: Simd> = T::Vector<S>;

    fn splat<S: Simd>(simd: S, rhs: Self::Rhs) -> Self::RhsVec<S> {
        simd.splat(rhs)
    }

    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: Self::RhsVec<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vbitor(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: Self::Rhs) -> T {
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

impl<T: VBitXor> ScalarSimdBinop<T, T> for VecBitXor {
    type Rhs = T;
    type RhsVec<S: Simd> = T::Vector<S>;

    fn splat<S: Simd>(simd: S, rhs: Self::Rhs) -> Self::RhsVec<S> {
        simd.splat(rhs)
    }

    fn apply_vec<S: Simd>(
        simd: S,
        lhs: <T as Vectorizable>::Vector<S>,
        rhs: Self::RhsVec<S>,
    ) -> <T as Vectorizable>::Vector<S> {
        T::vbitxor(simd, lhs, rhs)
    }

    fn apply(lhs: T, rhs: Self::Rhs) -> T {
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

pub fn try_binary_scalar_simd<
    E: NdArrayElement,
    EOut: NdArrayElement,
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: ScalarSimdBinop<T, Out>,
>(
    input: NdArrayTensor<E>,
    elem: Op::Rhs,
) -> Result<NdArrayTensor<EOut>, NdArrayTensor<E>> {
    if !should_use_simd(input.array.len()) || input.array.as_slice_memory_order().is_none() {
        return Err(input);
    }
    // Used to assert traits based on the dynamic `DType`.
    let input = unsafe { core::mem::transmute::<NdArrayTensor<E>, NdArrayTensor<T>>(input) };
    let out = if size_of::<T>() == size_of::<Out>() && input.array.is_unique() {
        binary_scalar_simd_inplace::<T, Out, Op>(input, elem)
    } else {
        binary_scalar_simd_owned::<T, Out, Op>(input, elem)
    };
    // Used to assert traits based on the dynamic `DType`.
    let out = unsafe { core::mem::transmute::<NdArrayTensor<Out>, NdArrayTensor<EOut>>(out) };
    Ok(out)
}

fn binary_scalar_simd_inplace<
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: ScalarSimdBinop<T, Out>,
>(
    input: NdArrayTensor<T>,
    elem: Op::Rhs,
) -> NdArrayTensor<Out> {
    let mut buffer = input.array.into_owned();
    let slice = buffer.as_slice_memory_order_mut().unwrap();
    let input = unsafe_alias_slice_mut(slice);
    // This is only called when in and out have the same size, so it's safe
    let out = unsafe { core::mem::transmute::<&mut [T], &mut [Out]>(slice) };
    binary_scalar::<T, Out, Op>(input, out, elem, PhantomData);
    // Buffer has the same elem size and is filled with the operation output, so this is safe
    let out = unsafe { core::mem::transmute::<ArrayD<T>, ArrayD<Out>>(buffer) };
    NdArrayTensor::new(out.into_shared())
}

fn binary_scalar_simd_owned<
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: ScalarSimdBinop<T, Out>,
>(
    input: NdArrayTensor<T>,
    elem: Op::Rhs,
) -> NdArrayTensor<Out> {
    let mut out = unsafe { ArrayD::uninit(input.array.shape()).assume_init() };
    let input = input.array.as_slice_memory_order().unwrap();
    let out_slice = out.as_slice_memory_order_mut().unwrap();
    binary_scalar::<T, Out, Op>(input, out_slice, elem, PhantomData);
    NdArrayTensor::new(out.into_shared())
}

#[pulp::with_simd(binary_scalar = Arch::new())]
fn binary_scalar_simd_slice<
    S: Simd,
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: ScalarSimdBinop<T, Out>,
>(
    simd: S,
    input: &[T],
    out: &mut [Out],
    elem: Op::Rhs,
    _op: PhantomData<Op>,
) {
    let (head, main, tail) = unsafe { input.align_to::<T::Vector<S>>() };
    let (out_head, out_main, out_tail) = unsafe { out.align_to_mut::<Out::Vector<S>>() };
    for (a, b) in head.iter().zip(out_head.iter_mut()) {
        *b = Op::apply(*a, elem);
    }
    for (a, b) in tail.iter().zip(out_tail.iter_mut()) {
        *b = Op::apply(*a, elem);
    }
    let elem_vec = Op::splat(simd, elem);
    let mut chunks = main.chunks_exact(4);
    let mut chunks_out = out_main.chunks_exact_mut(4);
    while let Some((a, b)) = chunks.next().zip(chunks_out.next()) {
        unsafe {
            let a_ptr = a as *const _ as *const T;
            let b_ptr = b as *mut _ as *mut Out;

            let s0 = simd.vload(a_ptr);
            let s1 = simd.vload(a_ptr.add(T::lanes::<S>()));
            let s2 = simd.vload(a_ptr.add(2 * T::lanes::<S>()));
            let s3 = simd.vload(a_ptr.add(3 * T::lanes::<S>()));

            let s0 = Op::apply_vec(simd, s0, elem_vec);
            let s1 = Op::apply_vec(simd, s1, elem_vec);
            let s2 = Op::apply_vec(simd, s2, elem_vec);
            let s3 = Op::apply_vec(simd, s3, elem_vec);

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

            let s0 = Op::apply_vec(simd, simd.vload(a_ptr), elem_vec);
            simd.vstore(b_ptr, s0);
        }
    }
}

fn unsafe_alias_slice_mut<'a, T>(slice: &mut [T]) -> &'a mut [T] {
    let ptr = slice.as_mut_ptr();
    let len = slice.len();
    unsafe { slice::from_raw_parts_mut(ptr, len) }
}
