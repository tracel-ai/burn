use burn_cube::{cube, CubeContext, Float};
use burn_jit::gpu::FloatKind::F32;
use burn_jit::gpu::{Elem, Item, Variable};

#[cube]
pub fn literal(lhs: Float) {
    let rhs: Float = 5.9f32.into();
}

#[test]
fn cube_literal_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Float(F32)));

    literal_expand(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref());
}

fn gpu_macro_ref() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Float(F32));

    let lhs = context.create_local(item);
    let lhs: Variable = lhs.into();
    let mut scope = context.into_scope();
    scope.create_with_value(5.9, item);

    format!("{:?}", scope.operations)
}
