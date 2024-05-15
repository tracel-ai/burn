use burn_cube::{branch::*, cube, CubeContext, Numeric, PrimitiveVariable, F32};
use burn_jit::{
    gpu,
    gpu::{Elem, Item, Variable},
};

type ElemType = F32;

#[cube]
pub fn if_greater<T: Numeric>(lhs: T) {
    if lhs > T::lit(0) {
        let _ = lhs + T::lit(4);
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

    gpu!(scope, cond = lhs > 0f32);
    gpu!(&mut scope, if(cond).then(|scope| {
        gpu!(scope, y = lhs + 4.0f32);
    }));

    format!("{:?}", scope.operations)
}
