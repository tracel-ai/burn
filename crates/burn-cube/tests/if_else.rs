use burn_cube::{branch::*, cube, elemtype::*, CubeContext, Float, F32};
use burn_jit::{
    cube_inline,
    gpu::{Elem, Item, Variable},
};

type ElemType = F32;

#[cube]
pub fn if_then_else<F: Float>(lhs: F) {
    if lhs < F::new(0.0) {
        let _ = lhs + F::new(4.0);
    } else {
        let _ = lhs - F::new(5.0);
    }
}

#[test]
fn cube_if_else_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Float(ElemType::into_kind())));

    if_then_else_expand::<ElemType>(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
}

fn inline_macro_ref() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Float(ElemType::into_kind()));
    let lhs = context.create_local(item);

    let mut scope = context.into_scope();
    let cond = scope.create_local(Item::Scalar(Elem::Bool));
    let lhs: Variable = lhs.into();
    let y = scope.create_local(item);

    cube_inline!(scope, cond = lhs < 0f32);
    cube_inline!(&mut scope, if(cond).then(|scope| {
        cube_inline!(scope, y = lhs + 4.0f32);
    }).else(|scope|{
        cube_inline!(scope, y = lhs - 5.0f32);
    }));

    format!("{:?}", scope.operations)
}
