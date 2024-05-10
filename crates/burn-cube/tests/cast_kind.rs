use burn_cube::{cube, elemtype::*, CubeContext, Float, Int, Numeric, F32, F64, I32, I64};
use burn_jit::{
    cube_inline,
    gpu::{Elem, Item},
};

#[cube]
pub fn cast_float_kind<F1: Float, F2: Float>(input: F1) {
    let x = input + F1::new(5.9);
    let y = to_float::<F1, F2>(x);
    let _ = y + F2::new(2.3);
}

#[cube]
pub fn cast_int_kind<I1: Int, I2: Int>(input: I1) {
    let x = input + I1::new(5.);
    let y = to_int::<I1, I2>(x);
    let _ = y + I2::new(2.);
}

#[cube]
pub fn cast_numeric_to_kind<T: Numeric, I2: Int>(input: T) {
    let x = input + T::new(5.);
    let y = to_int::<T, I2>(x);
    let _ = y + I2::new(2.);
}

#[test]
fn cube_cast_float_kind_test() {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Float(F64::into_kind()));

    let input = context.create_local(item);

    // F16 not testable with the gpu macro, but should work the same
    cast_float_kind_expand::<F64, F32>(&mut context, input);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_float());
}

#[test]
fn cube_cast_int_kind_test() {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Int(I32::into_kind()));

    let input = context.create_local(item);

    cast_int_kind_expand::<I32, I64>(&mut context, input);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
}

#[test]
fn cube_cast_numeric_kind_test() {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Int(I32::into_kind()));

    let input = context.create_local(item);

    cast_numeric_to_kind_expand::<I32, I64>(&mut context, input);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
}

fn inline_macro_ref_float() -> String {
    let mut context = CubeContext::root();
    let float_64 = Item::Scalar(Elem::Float(F64::into_kind()));
    let float_32 = Item::Scalar(Elem::Float(F32::into_kind()));
    let input = context.create_local(float_64);

    let mut scope = context.into_scope();
    let x = scope.create_local(float_64);
    let y = scope.create_local(float_32);
    let z = scope.create_local(float_32);

    cube_inline!(scope, x = input + 5.9f32 as f64);
    cube_inline!(scope, y = cast(x));
    cube_inline!(scope, z = y + 2.3f32);

    format!("{:?}", scope.operations)
}

fn inline_macro_ref_int() -> String {
    let mut context = CubeContext::root();
    let int_32 = Item::Scalar(Elem::Int(I32::into_kind()));
    let int_64 = Item::Scalar(Elem::Int(I64::into_kind()));
    let input = context.create_local(int_32);

    let mut scope = context.into_scope();
    let x = scope.create_local(int_32);
    let y = scope.create_local(int_64);
    let z = scope.create_local(int_64);

    cube_inline!(scope, x = input + 5i32);
    cube_inline!(scope, y = cast(x));
    cube_inline!(scope, z = y + 2i64);

    format!("{:?}", scope.operations)
}