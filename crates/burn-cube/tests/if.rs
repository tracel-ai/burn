use burn_cube::{cube, if_expand, CubeContext, Float};
use burn_jit::gpu;
use burn_jit::gpu::FloatKind::F32;
use burn_jit::gpu::{Elem, Item, Variable};

#[cube]
pub fn if_greater(lhs: Float) {
    if lhs > float_new(0.0) {
        let _ = lhs;
    }
}

#[test]
fn cube_if_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Float(F32)));

    if_greater_expand(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref());
}

fn gpu_macro_ref() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Float(F32));
    let lhs = context.create_local(item);

    let mut scope = context.into_scope();
    let cond = scope.create_local(Item::Scalar(Elem::Bool));
    let lhs: Variable = lhs.into();
    gpu!(scope, cond = lhs > 0f32);

    gpu!(&mut scope, if(cond).then(|_scope| {}));

    format!("{:?}", scope.operations)
}
