use burn_cube::{branch::*, cube, CubeContext, Int, Numeric, I32};
use burn_jit::{
    cube_inline,
    gpu::{Branch, Elem, Item, Variable},
};

// a += b is more efficient than a = a + b
// because the latter does not assume that a is the same in lhs and rhs
// It could be detected and optimized

type ElemType = I32;

#[cube]
pub fn reuse<I: Int>(mut x: I) {
    while x < I::new(10) {
        x = x + I::new(1);
    }
}

#[cube]
pub fn reuse_incr<I: Int>(mut x: I) {
    while x < I::new(10) {
        x += I::new(1);
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

    cube_inline!(
        &mut scope,
        loop(|scope| {
            cube_inline!(scope, cond = x < 10);
            cube_inline!(scope, if(cond).then(|scope|{
                scope.register(Branch::Break);
            }));

            cube_inline!(scope, tmp = x + 1);
            cube_inline!(scope, x = tmp);
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

    cube_inline!(
        &mut scope,
        loop(|scope| {
            cube_inline!(scope, cond = x < 10);
            cube_inline!(scope, if(cond).then(|scope|{
                scope.register(Branch::Break);
            }));

            cube_inline!(scope, x = x + 1);
        })
    );

    format!("{:?}", scope.operations)
}
