use burn_cube::{cube, CubeContext, Float, PrimitiveVariable, F32};
use burn_jit::{gpu, gpu::Item};

type ElemType = F32;

#[cube]
pub fn literal<F: Float>(lhs: F) {
    let _ = lhs + F::lit(5);
}

#[cube]
pub fn literal_float_no_decimals<F: Float>(lhs: F) {
    let _ = lhs + F::from_primitive(5.);
}

#[test]
fn cube_literal_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(ElemType::into_elem()));

    literal_expand::<ElemType>(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
}

#[test]
fn cube_literal_float_no_decimal_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(ElemType::into_elem()));

    literal_float_no_decimals_expand::<ElemType>(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
}

fn inline_macro_ref() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(ElemType::into_elem());
    let lhs = context.create_local(item);

    let mut scope = context.into_scope();
    let out = scope.create_local(item);
    gpu!(scope, out = lhs + 5.0f32);

    format!("{:?}", scope.operations)
}
