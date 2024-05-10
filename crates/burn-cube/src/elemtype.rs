use burn_jit::gpu::{Elem, Item};

use crate::{assign, Bool, CubeContext, CubeType, ExpandElement, Float, Int, UInt};

pub fn uint_new(val: u32) -> UInt {
    UInt {
        val,
        vectorization: 1,
    }
}
pub fn uint_new_expand(_context: &mut CubeContext, val: u32) -> <UInt as CubeType>::ExpandType {
    val.into()
}

pub fn bool_new(val: bool) -> Bool {
    Bool {
        val,
        vectorization: 1,
    }
}
pub fn bool_new_expand(_context: &mut CubeContext, val: bool) -> <Bool as CubeType>::ExpandType {
    val.into()
}

pub fn to_int<R: CubeType, I: Int>(_input: R) -> I {
    I::new(0)
}
pub fn to_int_expand<R: CubeType, I: Int>(
    context: &mut CubeContext,
    val: ExpandElement,
) -> <I as CubeType>::ExpandType {
    let new_var = context.create_local(Item::Scalar(I::into_elem()));
    assign::expand(context, val.into(), new_var.clone());
    new_var
}

pub fn to_float<R: CubeType, F: Float>(_input: R) -> F {
    // TODO: make val accessible through trait
    F::new(0)
}

pub fn to_float_expand<R: CubeType, F: Float>(
    context: &mut CubeContext,
    val: ExpandElement,
) -> ExpandElement {
    let new_var = context.create_local(Item::Scalar(F::into_elem()));
    assign::expand(context, val.into(), new_var.clone());
    new_var
}

pub fn to_uint<R: CubeType>(_input: R) -> UInt {
    UInt {
        val: 0,
        vectorization: 1,
    }
}
pub fn to_uint_expand(
    context: &mut CubeContext,
    val: ExpandElement,
) -> <UInt as CubeType>::ExpandType {
    let new_var = context.create_local(Item::Scalar(Elem::UInt));
    assign::expand(context, val.into(), new_var.clone());
    new_var
}

pub fn to_bool<R: CubeType>(_input: R) -> Bool {
    Bool {
        val: true,
        vectorization: 1,
    }
}
pub fn to_bool_expand(
    context: &mut CubeContext,
    val: ExpandElement,
) -> <UInt as CubeType>::ExpandType {
    let new_var = context.create_local(Item::Scalar(Elem::Bool));
    assign::expand(context, val.into(), new_var.clone());
    new_var
}
