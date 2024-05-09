use burn_cube::{cube, if_else_expand, CubeContext, Float, F32};
use burn_jit::gpu;
use burn_jit::gpu::{Elem, Item, Variable};

type ElemType = F32;

#[cube]
pub fn if_else<F: Float>(lhs: F) {
    if lhs < float_new::<F>(0.0) {
        let _ = lhs + float_new::<F>(4.0);
    } else {
        let _ = lhs - float_new::<F>(5.0);
    }
}

#[test]
fn cube_if_else_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Float(ElemType::into_kind())));

    if_else::expand::<ElemType>(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref());
}

fn gpu_macro_ref() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Float(ElemType::into_kind()));
    let lhs = context.create_local(item);

    let mut scope = context.into_scope();
    let cond = scope.create_local(Item::Scalar(Elem::Bool));
    let lhs: Variable = lhs.into();
    let y = scope.create_local(item);

    gpu!(scope, cond = lhs < 0f32);
    gpu!(&mut scope, if(cond).then(|scope| {
        gpu!(scope, y = lhs + 4.0f32);
    }).else(|scope|{
        gpu!(scope, y = lhs - 5.0f32);
    }));

    format!("{:?}", scope.operations)
}
