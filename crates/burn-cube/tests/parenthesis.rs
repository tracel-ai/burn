use burn_cube::{cube, CubeContext, Numeric, PrimitiveVariable, F32};
use burn_jit::{
    gpu,
    gpu::{Item, Variable},
};

type ElemType = F32;

#[cube]
pub fn parenthesis<T: Numeric>(x: T, y: T, z: T) -> T {
    x * (y + z)
}

#[test]
fn cube_parenthesis_priority_test() {
    let mut context = CubeContext::root();

    let x = context.create_local(Item::Scalar(ElemType::into_elem()));
    let y = context.create_local(Item::Scalar(ElemType::into_elem()));
    let z = context.create_local(Item::Scalar(ElemType::into_elem()));

    parenthesis_expand::<ElemType>(&mut context, x, y, z);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
}

fn inline_macro_ref() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(ElemType::into_elem());
    let x = context.create_local(item);
    let y = context.create_local(item);
    let z = context.create_local(item);

    let mut scope = context.into_scope();
    let x: Variable = x.into();
    let y: Variable = y.into();
    let z: Variable = z.into();
    let tmp = scope.create_local(item);

    gpu!(scope, tmp = y + z);
    gpu!(scope, y = x * tmp);

    format!("{:?}", scope.operations)
}
