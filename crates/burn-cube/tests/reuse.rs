use burn_cube::{cube, while_loop_expand, CubeContext, Int};
use burn_jit::gpu;
use burn_jit::gpu::IntKind::I32;
use burn_jit::gpu::{Branch, Elem, Item, Variable};

// TODO
// a += b is more efficient than a = a + b
// because the latter does not assume that a is the same in lhs and rhs
// It could be detected and optimized

#[cube]
pub fn reuse(mut x: Int) {
    while x < int_new(10) {
        x = x + int_new(1);
    }
}

#[cube]
pub fn reuse_incr(mut x: Int) {
    while x < int_new(10) {
        x += int_new(1);
    }
}

#[test]
fn cube_reuse_assign_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Int(I32)));

    reuse::expand(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref_assign());
}

#[test]
fn cube_reuse_incr_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Int(I32)));

    reuse_incr::expand(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref_incr());
}

fn gpu_macro_ref_assign() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Int(I32));
    let lhs = context.create_local(item);

    let mut scope = context.into_scope();
    let cond = scope.create_local(Item::Scalar(Elem::Bool));
    let lhs: Variable = lhs.into();
    let tmp = scope.create_local(item);

    gpu!(
        &mut scope,
        loop(|scope| {
            gpu!(scope, cond = lhs < 10);
            gpu!(scope, if(cond).then(|scope|{
                scope.register(Branch::Break);
            }));

            gpu!(scope, tmp = lhs + 1);
            gpu!(scope, lhs = tmp);
        })
    );

    format!("{:?}", scope.operations)
}

fn gpu_macro_ref_incr() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Int(I32));
    let lhs = context.create_local(item);

    let mut scope = context.into_scope();
    let cond = scope.create_local(Item::Scalar(Elem::Bool));
    let lhs: Variable = lhs.into();

    gpu!(
        &mut scope,
        loop(|scope| {
            gpu!(scope, cond = lhs < 10);
            gpu!(scope, if(cond).then(|scope|{
                scope.register(Branch::Break);
            }));

            gpu!(scope, lhs = lhs + 1);
        })
    );

    format!("{:?}", scope.operations)
}
