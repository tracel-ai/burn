use burn_cube::{cube, CubeContext, Numeric, PrimitiveVariable, F32, I32};
use burn_jit::{gpu, gpu::Item};

#[cube]
pub fn generic_kernel<T: Numeric>(lhs: T) {
    let _ = lhs + T::lit(5);
}

#[test]
fn cube_generic_float_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(F32::into_elem()));

    generic_kernel_expand::<F32>(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_float());
}

#[test]
fn cube_generic_int_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(I32::into_elem()));

    generic_kernel_expand::<I32>(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
}

fn inline_macro_ref_float() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(F32::into_elem());
    let lhs = context.create_local(item);

    let mut scope = context.into_scope();
    let out = scope.create_local(item);
    gpu!(scope, out = lhs + 5.0f32);

    format!("{:?}", scope.operations)
}

fn inline_macro_ref_int() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(I32::into_elem());
    let lhs = context.create_local(item);

    let mut scope = context.into_scope();
    let out = scope.create_local(item);
    gpu!(scope, out = lhs + 5);

    format!("{:?}", scope.operations)
}
