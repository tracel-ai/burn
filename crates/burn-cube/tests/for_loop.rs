use burn_cube::{cube, range, range_expand, Array, CubeContext, Float, UInt};
use burn_jit::gpu;
use burn_jit::gpu::FloatKind::F32;
use burn_jit::gpu::{Elem, Item, Variable};

#[cube]
pub fn kernel(mut lhs: Array<Float>, rhs: Float, end: UInt, unroll: bool) {
    let tmp1 = rhs * rhs;
    let tmp2 = tmp1 + rhs;

    for i in range(0u32, end, unroll) {
        lhs[i] = tmp2 + lhs[i];
    }
}

// #[allow(unused_mut)]
// pub fn kernel_expand(
//     context: &mut burn_cube::CubeContext,
//     mut lhs: <Array<Float> as burn_cube::RuntimeType>::ExpandType,
//     rhs: <Float as burn_cube::RuntimeType>::ExpandType,
//     end: <UInt as burn_cube::RuntimeType>::ExpandType,
//     unroll: <bool as burn_cube::RuntimeType>::ExpandType,
// ) -> () {
//     let tmp1 = {
//         let _lhs = rhs.clone().into();
//         let _rhs = rhs.clone().into();
//         burn_cube::mul::expand(context, _lhs, _rhs)
//     };
//     let tmp2 = {
//         let _lhs = tmp1.into();
//         let _rhs = rhs.into();
//         burn_cube::add::expand(context, _lhs, _rhs)
//     };
//     range_expand(
//         context,
//         0u32.into(),
//         end.into(),
//         unroll.into(),
//         |context, i| {
//             {
//                 let _array = lhs.clone().into();
//                 let _index = i.clone().into();
//                 let _value = {
//                     let _lhs = tmp2.clone().into();
//                     let _rhs = {
//                         let _array = lhs.clone().into();
//                         let _index = i.into();
//                         burn_cube::index::expand(context, _array, _index)
//                     };
//                     burn_cube::add::expand(context, _lhs, _rhs)
//                 };
//                 burn_cube::index_assign::expand(context, _array, _index, _value)
//             };
//         },
//     );
// }

#[test]
fn test_for_loop_with_unroll() {
    let mut context = CubeContext::root();
    let unroll = true;

    let lhs = context.create_local(Item::Scalar(Elem::Float(F32)));
    let rhs = context.create_local(Item::Scalar(Elem::Float(F32)));
    let end = 4u32.into();

    kernel_expand(&mut context, lhs, rhs, end, unroll);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref(unroll));
}

#[test]
fn test_for_loop_no_unroll() {
    let mut context = CubeContext::root();
    let unroll = false;

    let lhs = context.create_local(Item::Scalar(Elem::Float(F32)));
    let rhs = context.create_local(Item::Scalar(Elem::Float(F32)));
    let end = 4u32.into();

    kernel_expand(&mut context, lhs, rhs, end, unroll);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref(unroll));
}

fn gpu_macro_ref(unroll: bool) -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Float(F32));

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
