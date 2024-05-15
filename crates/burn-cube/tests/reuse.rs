use burn_cube::{branch::*, cube, CubeContext, Int, PrimitiveVariable, I32};
use burn_jit::{
    gpu,
    gpu::{Branch, Elem, Item, Variable},
};

type ElemType = I32;

#[cube]
#[allow(clippy::assign_op_pattern)]
pub fn reuse<I: Int>(mut x: I) {
    // a += b is more efficient than a = a + b
    // Because the latter does not assume that a is the same in lhs and rhs
    // Normally clippy should detect it
    while x < I::lit(10) {
        x = x + I::lit(1);
    }
}

#[cube]
pub fn reuse_incr<I: Int>(mut x: I) {
    while x < I::lit(10) {
        x += I::lit(1);
    }
}

#[test]
fn cube_reuse_assign_test() {
    let mut context = CubeContext::root();

    let x = context.create_local(Item::Scalar(ElemType::into_elem()));

    reuse_expand::<ElemType>(&mut context, x);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_assign());
}

#[test]
fn cube_reuse_incr_test() {
    let mut context = CubeContext::root();

    let x = context.create_local(Item::Scalar(ElemType::into_elem()));

    reuse_incr_expand::<ElemType>(&mut context, x);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_incr());
}

fn inline_macro_ref_assign() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(ElemType::into_elem());
    let x = context.create_local(item);

    let mut scope = context.into_scope();
    let cond = scope.create_local(Item::Scalar(Elem::Bool));
    let x: Variable = x.into();
    let tmp = scope.create_local(item);

    gpu!(
        &mut scope,
        loop(|scope| {
            gpu!(scope, cond = x < 10);
            gpu!(scope, if(cond).then(|scope|{
                scope.register(Branch::Break);
            }));

            gpu!(scope, tmp = x + 1);
            gpu!(scope, x = tmp);
        })
    );

    format!("{:?}", scope.operations)
}

fn inline_macro_ref_incr() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(ElemType::into_elem());
    let x = context.create_local(item);

    let mut scope = context.into_scope();
    let cond = scope.create_local(Item::Scalar(Elem::Bool));
    let x: Variable = x.into();

    gpu!(
        &mut scope,
        loop(|scope| {
            gpu!(scope, cond = x < 10);
            gpu!(scope, if(cond).then(|scope|{
                scope.register(Branch::Break);
            }));

            gpu!(scope, x = x + 1);
        })
    );

    format!("{:?}", scope.operations)
}
