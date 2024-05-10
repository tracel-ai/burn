use burn_jit::gpu::{Elem, Item};

use crate::{assign, Bool, CubeContext, CubeType, ExpandElement, Float, Int, Numeric, UInt};

// pub fn new<T: Numeric>(val: f) -> T {
//     T::new(val)
// }

// pub fn new_expand<T: Numeric>(_context: &mut CubeContext, val: f32) -> <T as CubeType>::ExpandType {
//     T::new_expand(_context, val)
// }

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
    I::new(0.)
}
pub fn to_int_expand<R: CubeType, I: Int>(
    context: &mut CubeContext,
    val: ExpandElement,
) -> <I as CubeType>::ExpandType {
    let elem = Elem::Int(I::into_kind());
    let new_var = context.create_local(match val.item() {
        Item::Vec4(_) => Item::Vec4(elem),
        Item::Vec3(_) => Item::Vec3(elem),
        Item::Vec2(_) => Item::Vec2(elem),
        Item::Scalar(_) => Item::Scalar(elem),
    });
    assign::expand(context, val.into(), new_var.clone());
    new_var
}

pub fn to_float<R: CubeType, F: Float>(_input: R) -> F {
    // TODO: make val accessible through trait
    F::new(0.)
}

pub fn to_float_expand<R: CubeType, F: Float>(
    context: &mut CubeContext,
    val: ExpandElement,
) -> ExpandElement {
    let elem = Elem::Float(F::into_kind());
    let new_var = context.create_local(match val.item() {
        Item::Vec4(_) => Item::Vec4(elem),
        Item::Vec3(_) => Item::Vec3(elem),
        Item::Vec2(_) => Item::Vec2(elem),
        Item::Scalar(_) => Item::Scalar(elem),
    });
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
    let elem = Elem::UInt;
    let new_var = context.create_local(match val.item() {
        Item::Vec4(_) => Item::Vec4(elem),
        Item::Vec3(_) => Item::Vec3(elem),
        Item::Vec2(_) => Item::Vec2(elem),
        Item::Scalar(_) => Item::Scalar(elem),
    });
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
    let elem = Elem::Bool;
    let new_var = context.create_local(match val.item() {
        Item::Vec4(_) => Item::Vec4(elem),
        Item::Vec3(_) => Item::Vec3(elem),
        Item::Vec2(_) => Item::Vec2(elem),
        Item::Scalar(_) => Item::Scalar(elem),
    });
    assign::expand(context, val.into(), new_var.clone());
    new_var
}
