use burn_cube::{branch::*, cube, CubeContext, Float, PrimitiveVariable, F32};
use burn_jit::{
    gpu,
    gpu::{Elem, Item, Variable},
};

type ElemType = F32;

#[cube]
pub fn if_then_else<F: Float>(lhs: F) {
    if lhs < F::lit(0) {
        let _ = lhs + F::lit(4);
    } else {
        let _ = lhs - F::lit(5);
    }
}

#[test]
fn cube_if_else_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(ElemType::into_elem()));

    if_then_else_expand::<ElemType>(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
}

fn inline_macro_ref() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(ElemType::into_elem());
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
