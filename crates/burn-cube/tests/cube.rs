use burn_cube::{cube, range, range_expand, Array, CubeContext, Float, UInt};
use burn_jit::gpu::{Elem, Item};

#[cube]
pub fn kernel(mut lhs: Array<Float>, rhs: Float, end: UInt, unroll: bool) {
    for i in range(0usize, end, unroll) {
        lhs[i] = rhs.clone() + lhs[i].clone();
    }
}

// #[cube]
// pub fn kernel_inner(lhs: Float, rhs: Float) -> Float {
//     lhs + rhs
// }

#[test]
fn test_simple_add() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Vec4(Elem::Float));
    let rhs = context.create_local(Item::Vec4(Elem::Float));
    let end = context.create_local(Item::Scalar(Elem::UInt));

    kernel_expand(&mut context, lhs, rhs, end, false);
}
