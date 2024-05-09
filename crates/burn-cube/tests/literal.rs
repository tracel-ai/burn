use burn_cube::{cube, CubeContext, Float, F32};
use burn_jit::{cube_inline, gpu::{Elem, Item}};

type ElemType = F32;

#[cube]
pub fn literal<F: Float>(lhs: F) {
    let _ = lhs + float_new::<F>(5.9);
}

#[test]
fn cube_literal_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Float(ElemType::into_kind())));

    literal::expand::<ElemType>(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
}

fn inline_macro_ref() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Float(ElemType::into_kind()));
    let lhs = context.create_local(item);

    let mut scope = context.into_scope();
    let out = scope.create_local(item);
    cube_inline!(scope, out = lhs + 5.9f32);

    format!("{:?}", scope.operations)
}
