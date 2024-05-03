// use burn_cube::{cube, while_loop_expand, CubeContext, Int};
// use burn_jit::gpu;
// use burn_jit::gpu::Branch;
// use burn_jit::gpu::IntKind::I32;
// use burn_jit::gpu::{Elem, Item, Variable};

// #[cube]
// pub fn while_not(lhs: Int) {
//     while lhs != int_new(0) {
//         let _ = lhs - int_new(1);
//     }
// }

// #[test]
// fn cube_while_test() {
//     let mut context = CubeContext::root();

//     let lhs = context.create_local(Item::Scalar(Elem::Int(I32)));

//     while_not_expand(&mut context, lhs);
//     let scope = context.into_scope();

//     assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref());
// }

// fn gpu_macro_ref() -> String {
//     let mut context = CubeContext::root();
//     let item = Item::Scalar(Elem::Int(I32));
//     let lhs = context.create_local(item);

//     let mut scope = context.into_scope();
//     let cond = scope.create_local(Item::Scalar(Elem::Bool));
//     let lhs: Variable = lhs.into();
//     let rhs = scope.create_local(item);

//     gpu!(
//         &mut scope,
//         loop(|scope| {
//             gpu!(scope, cond = lhs != 0);
//             gpu!(scope, if(cond).then(|scope|{
//                 scope.register(Branch::Break);
//             }));

//             gpu!(scope, rhs = lhs - 1i32);
//         })
//     );

//     format!("{:?}", scope.operations)
// }
