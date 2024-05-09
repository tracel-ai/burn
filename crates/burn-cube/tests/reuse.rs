use burn_cube::{cube, while_loop_expand, CubeContext, Int, I32};
use burn_jit::gpu;
use burn_jit::gpu::{Branch, Elem, Item, Variable};

// TODO
// a += b is more efficient than a = a + b
// because the latter does not assume that a is the same in lhs and rhs
// It could be detected and optimized

type ElemType = I32;

#[cube]
pub fn reuse<I: Int>(mut x: I) {
    while x < int_new::<I>(10) {
        x = x + int_new::<I>(1);
    }
}

#[cube]
pub fn reuse_incr<I: Int>(mut x: I) {
    while x < int_new::<I>(10) {
        x += int_new::<I>(1);
    }
}

#[test]
fn cube_reuse_assign_test() {
    let mut context = CubeContext::root();

    let x = context.create_local(Item::Scalar(Elem::Int(ElemType::into_kind())));

    reuse::expand::<ElemType>(&mut context, x);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref_assign());
}

#[test]
fn cube_reuse_incr_test() {
    let mut context = CubeContext::root();

    let x = context.create_local(Item::Scalar(Elem::Int(ElemType::into_kind())));

    reuse_incr::expand::<ElemType>(&mut context, x);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref_incr());
}

fn gpu_macro_ref_assign() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Int(ElemType::into_kind()));
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

fn gpu_macro_ref_incr() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Int(ElemType::into_kind()));
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
