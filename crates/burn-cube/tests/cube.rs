use burn_cube::{cube, range, range_expand, Array, CubeContext, Float, UInt};
use burn_jit::gpu::{Elem, Item};

#[cube]
pub fn kernel(mut lhs: Array<Float>, rhs: Float, end: UInt, unroll: bool) {
    let tmp1 = rhs * rhs;
    let tmp2 = tmp1 + rhs;

    for i in range(0usize, end, unroll) {
        lhs[i] = tmp2 + lhs[i];
    }
}

#[test]
fn test_simple_add() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Float));
    let rhs = context.create_local(Item::Scalar(Elem::Float));
    let end = context.create_local(Item::Scalar(Elem::UInt));

    kernel_expand(&mut context, lhs, rhs, end, false);
    let scope = context.into_scope();

    for op in scope.operations.iter() {
        println!("{op:?}");
    }

    panic!("nop");
}
