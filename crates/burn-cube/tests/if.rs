use burn_cube::{branch::*, cube, CubeContext, Float, Numeric, F32};
use burn_jit::{
    cube_inline,
    gpu::{Elem, Item, Variable},
};

type ElemType = F32;

#[cube]
pub fn if_greater<F: Float>(lhs: F) {
    if lhs > F::new(0) {
        let _ = lhs + F::new(4);
    }
}

#[test]
fn cube_if_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(ElemType::into_elem()));

    if_greater_expand::<ElemType>(&mut context, lhs);
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

    cube_inline!(scope, cond = lhs > 0f32);
    cube_inline!(&mut scope, if(cond).then(|scope| {
        cube_inline!(scope, y = lhs + 4.0f32);
    }));

    format!("{:?}", scope.operations)
}
