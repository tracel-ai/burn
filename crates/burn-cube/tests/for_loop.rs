use burn_cube::{branch::*, cube, Array, CubeContext, Float, PrimitiveVariable, UInt, F32};
use burn_jit::{
    gpu,
    gpu::{Item, Variable},
};

type ElemType = F32;

#[cube]
pub fn for_loop<F: Float>(mut lhs: Array<F>, rhs: F, end: UInt, unroll: bool) {
    let tmp1 = rhs * rhs;
    let tmp2 = tmp1 + rhs;

    for i in range(0u32, end, unroll) {
        lhs[i] = tmp2 + lhs[i];
    }
}

#[test]
fn test_for_loop_with_unroll() {
    let mut context = CubeContext::root();
    let unroll = true;

    let lhs = context.create_local(Item::Scalar(ElemType::into_elem()));
    let rhs = context.create_local(Item::Scalar(ElemType::into_elem()));
    let end = 4u32.into();

    for_loop_expand::<ElemType>(&mut context, lhs, rhs, end, unroll);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref(unroll));
}

#[test]
fn test_for_loop_no_unroll() {
    let mut context = CubeContext::root();
    let unroll = false;

    let lhs = context.create_local(Item::Scalar(ElemType::into_elem()));
    let rhs = context.create_local(Item::Scalar(ElemType::into_elem()));
    let end = 4u32.into();

    for_loop_expand::<ElemType>(&mut context, lhs, rhs, end, unroll);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref(unroll));
}

fn inline_macro_ref(unroll: bool) -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(ElemType::into_elem());

    let lhs = context.create_local(item);
    let rhs = context.create_local(item);
    let lhs: Variable = lhs.into();
    let rhs: Variable = rhs.into();
    let end = 4u32;
    let mut scope = context.into_scope();

    // Kernel
    let tmp1 = scope.create_local(item);
    let tmp2 = scope.create_local(item);
    gpu!(scope, tmp1 = rhs * rhs);
    gpu!(scope, tmp2 = tmp1 + rhs);

    gpu!(
        &mut scope,
        range(0u32, end, unroll).for_each(|i, scope| {
            gpu!(scope, rhs = lhs[i]);
            gpu!(scope, tmp1 = tmp2 + rhs);
            gpu!(scope, lhs[i] = tmp1);
        })
    );

    format!("{:?}", scope.operations)
}
