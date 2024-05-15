use burn_cube::{cube, CubeContext, Float, Int, Numeric, PrimitiveVariable, F32, F64, I32, I64};
use burn_jit::{gpu, gpu::Item};

#[cube]
pub fn cast_float_kind<F1: Float, F2: Float + From<F1>>(input: F1) {
    let x = input + F1::from_primitive(5.9);
    let y = F2::from(x);
    let _ = y + F2::from_primitive(2.3);
}

#[cube]
pub fn cast_int_kind<I1: Int, I2: Int + From<I1>>(input: I1) {
    let x = input + I1::from_primitive(5);
    let y = I2::from(x);
    let _ = y + I2::from_primitive(2);
}

#[cube]
pub fn cast_numeric_to_kind<T: Numeric, I2: Int + From<T>>(input: T) {
    let x = input + T::lit(5);
    let y = I2::from(x);
    let _ = y + I2::lit(2);
}

#[test]
fn cube_cast_float_kind_test() {
    let mut context = CubeContext::root();
    let item = Item::Scalar(F64::into_elem());

    let input = context.create_local(item);

    // F16 not testable with the gpu macro, but should work the same
    cast_float_kind_expand::<F64, F32>(&mut context, input);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_float());
}

#[test]
fn cube_cast_int_kind_test() {
    let mut context = CubeContext::root();
    let item = Item::Scalar(I32::into_elem());

    let input = context.create_local(item);

    cast_int_kind_expand::<I32, I64>(&mut context, input);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
}

#[test]
fn cube_cast_numeric_kind_test() {
    let mut context = CubeContext::root();
    let item = Item::Scalar(I32::into_elem());

    let input = context.create_local(item);

    cast_numeric_to_kind_expand::<I32, I64>(&mut context, input);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
}

fn inline_macro_ref_float() -> String {
    let mut context = CubeContext::root();
    let float_64 = Item::Scalar(F64::into_elem());
    let float_32 = Item::Scalar(F32::into_elem());
    let input = context.create_local(float_64);

    let mut scope = context.into_scope();
    let x = scope.create_local(float_64);
    let y = scope.create_local(float_32);
    let z = scope.create_local(float_32);

    gpu!(scope, x = input + 5.9f32 as f64);
    gpu!(scope, y = cast(x));
    gpu!(scope, z = y + 2.3f32);

    format!("{:?}", scope.operations)
}

fn inline_macro_ref_int() -> String {
    let mut context = CubeContext::root();
    let int_32 = Item::Scalar(I32::into_elem());
    let int_64 = Item::Scalar(I64::into_elem());
    let input = context.create_local(int_32);

    let mut scope = context.into_scope();
    let x = scope.create_local(int_32);
    let y = scope.create_local(int_64);
    let z = scope.create_local(int_64);

    gpu!(scope, x = input + 5i32);
    gpu!(scope, y = cast(x));
    gpu!(scope, z = y + 2i64);

    format!("{:?}", scope.operations)
}
