use core::marker::PhantomData;

use macerator::{SimdExt, VAdd, VBitAnd, VBitOr, VBitXor, VDiv, VMul, VOrd, VSub, Vectorizable};
use ndarray::ArrayD;
use pulp::{cast, Arch, Simd};

use crate::{NdArrayElement, NdArrayTensor};

use super::{load4, load4_unaligned, should_use_simd, store4, store4_unaligned, MinMax};

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
    binary_scalar_slice_inplace::<T, Out, Op>(slice, elem, PhantomData);
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
    binary_scalar_slice::<T, Out, Op>(input, out_slice, elem, PhantomData);
    NdArrayTensor::new(out.into_shared())
}

#[pulp::with_simd(binary_scalar_slice = Arch::new())]
fn binary_scalar_simd_slice<
    S: Simd,
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: ScalarSimdBinop<T, Out>,
>(
    simd: S,
    input: &[T],
    out: &mut [Out],
    rhs: Op::Rhs,
    _op: PhantomData<Op>,
) {
    let lanes = T::lanes::<S>();
    let mut chunks_input = input.chunks_exact(4 * lanes);
    let mut chunks_out = out.chunks_exact_mut(4 * lanes);
    let rhs_vec = Op::splat(simd, rhs);
    while let Some((input, out)) = chunks_input.next().zip(chunks_out.next()) {
        unsafe {
            let (s0, s1, s2, s3) = load4_unaligned(simd, input.as_ptr());

            let s0 = Op::apply_vec(simd, s0, rhs_vec);
            let s1 = Op::apply_vec(simd, s1, rhs_vec);
            let s2 = Op::apply_vec(simd, s2, rhs_vec);
            let s3 = Op::apply_vec(simd, s3, rhs_vec);

            store4_unaligned(simd, out.as_mut_ptr(), s0, s1, s2, s3);
        }
    }
    let mut chunks_input = chunks_input.remainder().chunks_exact(lanes);
    let mut chunks_out = chunks_out.into_remainder().chunks_exact_mut(lanes);
    while let Some((input, out)) = chunks_input.next().zip(chunks_out.next()) {
        unsafe {
            let s0 = simd.vload_unaligned(input as *const _ as *const T);
            let s0 = Op::apply_vec(simd, s0, rhs_vec);

            simd.vstore_unaligned(out as *mut _ as *mut Out, s0);
        }
    }

    for (input, out) in chunks_input
        .remainder()
        .iter()
        .zip(chunks_out.into_remainder())
    {
        *out = Op::apply(*input, rhs)
    }
}

#[pulp::with_simd(binary_scalar_slice_inplace = Arch::new())]
fn binary_scalar_simd_slice_inplace<
    S: Simd,
    T: NdArrayElement + Vectorizable,
    Out: NdArrayElement + Vectorizable,
    Op: ScalarSimdBinop<T, Out>,
>(
    simd: S,
    buf: &mut [T],
    rhs: Op::Rhs,
    _op: PhantomData<(Out, Op)>,
) {
    let (head, main, tail) = unsafe { buf.align_to_mut::<T::Vector<S>>() };
    for elem in head.iter_mut().chain(tail) {
        *elem = cast(Op::apply(*elem, rhs));
    }
    let mut chunks = main.chunks_exact_mut(4);
    let rhs = Op::splat(simd, rhs);
    for elem in chunks.by_ref() {
        unsafe {
            let (s0, s1, s2, s3) = load4(simd, elem as *const _ as *const T);

            let s0 = Op::apply_vec(simd, s0, rhs);
            let s1 = Op::apply_vec(simd, s1, rhs);
            let s2 = Op::apply_vec(simd, s2, rhs);
            let s3 = Op::apply_vec(simd, s3, rhs);

            store4(simd, elem as *mut _ as *mut Out, s0, s1, s2, s3);
        }
    }

    for elem in chunks.into_remainder() {
        let input_ptr = elem as *const _ as *const T;
        let out_ptr = elem as *mut _ as *mut Out;

        let s0 = unsafe { simd.vload(input_ptr) };

        let s0 = Op::apply_vec(simd, s0, rhs);

        unsafe { simd.vstore(out_ptr, s0) };
    }
}
