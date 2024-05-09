use burn_cube::{
    break_expand, cube, if_expand, loop_expand, while_loop_expand, CubeContext, Int, I32,
};
use burn_jit::cube_inline;
use burn_jit::gpu::Branch;
use burn_jit::gpu::{Elem, Item, Variable};

type ElemType = I32;

#[cube]
pub fn while_not<I: Int>(lhs: I) {
    while lhs != int_new::<I>(0) {
        let _ = lhs - int_new::<I>(1);
    }
}

#[cube]
pub fn manual_loop_break<I: Int>(lhs: I) {
    loop {
        if lhs != int_new::<I>(0) {
            break;
        }
        let _ = lhs - int_new::<I>(1);
    }
}

#[test]
fn cube_while_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Int(ElemType::into_kind())));

    while_not::expand::<ElemType>(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
}

#[test]
fn cube_loop_break_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Int(ElemType::into_kind())));

    manual_loop_break::expand::<ElemType>(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
}

fn inline_macro_ref() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Int(ElemType::into_kind()));
    let lhs = context.create_local(item);

    let mut scope = context.into_scope();
    let cond = scope.create_local(Item::Scalar(Elem::Bool));
    let lhs: Variable = lhs.into();
    let rhs = scope.create_local(item);

    cube_inline!(
        &mut scope,
        loop(|scope| {
            cube_inline!(scope, cond = lhs != 0);
            cube_inline!(scope, if(cond).then(|scope|{
                scope.register(Branch::Break);
            }));

            cube_inline!(scope, rhs = lhs - 1i32);
        })
    );

    format!("{:?}", scope.operations)
}
