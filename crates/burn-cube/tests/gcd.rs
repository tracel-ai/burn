// use burn_cube::cube;
// use burn_jit::gpu;
// use burn_jit::gpu::Branch;
// use burn_jit::gpu::IntKind::I32;
// use burn_jit::gpu::{Elem, Item, Variable};

// #[cube]
// pub fn gcd(lhs: Int, rhs: Int) {
//     while rhs != int_new(0) {
//         let tmp = rhs;
//         rhs = lhs % rhs;
//         lhs = tmp
//     }
//     // TODO: use lhs as output
// }

// #[test]
// fn cube_function_test() {
//     let mut context = CubeContext::root();

//     let lhs = context.create_local(Item::Scalar(Elem::Int(I32)));
//     let rhs = context.create_local(Item::Scalar(Elem::Int(I32)));

//     gcd_expand(&mut context, lhs, rhs);
//     let scope = context.into_scope();

//     assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref());
// }

// fn gpu_macro_ref() -> String {
//     let mut context = CubeContext::root();
//     let item = Item::Scalar(Elem::Int(I32));

//     let lhs = context.create_local(item);
//     let rhs = context.create_local(item);
//     let lhs: Variable = lhs.into();
//     let rhs: Variable = rhs.into();
//     let mut scope = context.into_scope();

//     // Kernel
//     let cond = scope.create_local(Item::Scalar(Elem::Bool));
//     let tmp = scope.create_local(Item::Scalar(Elem::Int(I32)));
//     gpu!(
//         &mut scope,
//         loop(|scope| {
//             gpu!(scope, cond = rhs != 0);
//             gpu!(scope, if(cond).then(|scope|{
//                 scope.register(Branch::Break);
//             }));

//             gpu!(scope, tmp = rhs);
//             gpu!(scope, rhs = lhs % rhs);
//             gpu!(scope, lhs = tmp);
//         })
//     );

//     format!("{:?}", scope.operations)
// }
