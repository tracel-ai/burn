use core::marker::PhantomData;

use macerator::{SimdExt, VAdd, VBitAnd, VBitOr, VBitXor, VDiv, VMul, VOrd, VSub, Vectorizable};
use ndarray::ArrayD;
use pulp::{cast, Arch, Simd};
use seq_macro::seq;

use crate::{NdArrayElement, NdArrayTensor};

use super::{should_use_simd, MinMax};

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
    let out = if size_of::<T>() == size_of::<Out>()
        && align_of::<T>() >= align_of::<Out>()
        && input.array.is_unique()
    {
        unsafe { binary_scalar_simd_inplace::<T, Out, Op>(input, elem) }
    } else {
        binary_scalar_simd_owned::<T, Out, Op>(input, elem)
    };
    // Used to assert traits based on the dynamic `DType`.
    let out = unsafe { core::mem::transmute::<NdArrayTensor<Out>, NdArrayTensor<EOut>>(out) };
    Ok(out)
}

/// Execute operation in place on an owned tensor
/// SAFETY:
/// Must ensure `size_of::<T> == size_of::<Out>` and `align_of::<T> >= align_of::<Out>`.
unsafe fn binary_scalar_simd_inplace<
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

/// Create a new copy of the tensor as the output
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

#[inline(always)]
#[allow(clippy::erasing_op, clippy::identity_op)]
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
    let mut chunks_input = input.chunks_exact(8 * lanes);
    let mut chunks_out = out.chunks_exact_mut(8 * lanes);
    let rhs_vec = Op::splat(simd, rhs);
    while let Some((input, out)) = chunks_input.next().zip(chunks_out.next()) {
        seq!(N in 0..8 {
            // Load one full vector from `input`.
            // SAFETY: Guaranteed to be in bounds because `len == 8 * lanes`
            let s~N = unsafe { simd.vload_unaligned(&input[N * lanes]) };
            let s~N = Op::apply_vec(simd, s~N, rhs_vec);
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
        let s0 = Op::apply_vec(simd, s0, rhs_vec);
        // Store one full vector to `out`.
        // SAFETY: Guaranteed to be in bounds because `len == lanes`
        unsafe { simd.vstore_unaligned(out.as_mut_ptr(), s0) };
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
#[pulp::with_simd(binary_scalar_slice_inplace = Arch::new())]
unsafe fn binary_scalar_simd_slice_inplace<
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
    let mut chunks = main.chunks_exact_mut(8);
    let rhs = Op::splat(simd, rhs);
    for elem in chunks.by_ref() {
        seq!(N in 0..8 {
            // Load a full vector from the aligned portion of the buffer.
            // SAFETY: `align_to_mut` guarantees we're aligned to `T::Vector`'s size, and there is
            // always a full vector in bounds.
            let s~N = unsafe { simd.vload(&elem[N] as *const _ as *const T) };
            let s~N = Op::apply_vec(simd, s~N, rhs);
            // Store a full vector at the same position as the input. Cast is safe because `Out` is
            // size and align compatible
            unsafe { simd.vstore_unaligned(&mut elem[N] as *mut _ as *mut Out, s~N) };
        });
    }

    for elem in chunks.into_remainder() {
        // Load a full vector from the aligned portion of the buffer.
        // SAFETY: `align_to_mut` guarantees we're aligned to `T::Vector`'s size, and there is
        // always a full vector in bounds.
        let s0 = unsafe { simd.vload(elem as *const _ as *const T) };

        let s0 = Op::apply_vec(simd, s0, rhs);
        // Store a full vector at the same position as the input. Cast is safe because `Out` is
        // size and align compatible
        unsafe { simd.vstore(elem as *mut _ as *mut Out, s0) };
    }
}
