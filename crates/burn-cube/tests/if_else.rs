use burn_cube::{cube, if_else_expand, CubeContext, Float};
use burn_jit::gpu;
use burn_jit::gpu::FloatKind::F32;
use burn_jit::gpu::{Elem, Item, Variable};

#[cube]
pub fn if_else(lhs: Float) {
    if lhs < float_new(0.0) {
        let _ = lhs + float_new(4.0);
    } else {
        let _ = lhs - float_new(5.0);
    }
}

#[test]
fn cube_if_else_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Float(F32)));

    if_else::expand(&mut context, lhs);
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
    let y = scope.create_local(item);

    gpu!(scope, cond = lhs < 0f32);
    gpu!(&mut scope, if(cond).then(|scope| {
        gpu!(scope, y = lhs + 4.0);
    }).else(|scope|{
        gpu!(scope, y = lhs - 5.0);
    }));

    format!("{:?}", scope.operations)
}
