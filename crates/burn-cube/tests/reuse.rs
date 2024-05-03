use burn_cube::{cube, while_loop_expand, CubeContext, Int};
use burn_jit::gpu;
use burn_jit::gpu::IntKind::I32;
use burn_jit::gpu::{Branch, Elem, Item, Variable};

// #[cube]
// pub fn reuse(lhs: Int) {
//     while lhs < int_new(10) {
//         lhs = lhs + int_new(1);
//     }
// }

#[cube]
pub fn reuse_incr(mut lhs: Int) {
    while lhs < int_new(10) {
        lhs += int_new(1);
    }
}

// #[test]
// fn cube_reuse_test() {
//     let mut context = CubeContext::root();

//     let lhs = context.create_local(Item::Scalar(Elem::Int(I32)));

//     reuse::expand(&mut context, lhs);
//     let scope = context.into_scope();

//     assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref());
// }

#[test]
fn cube_reuse_incr_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Int(I32)));

    reuse_incr::expand(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref());
}

fn gpu_macro_ref() -> String {
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
