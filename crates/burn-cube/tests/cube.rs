use burn_cube::{cube, CubeContext, Float};
use burn_jit::gpu::{Elem, Item};

#[cube]
pub fn kernel(lhs: Float, rhs: Float) -> Float {
    let out = lhs.clone() + rhs;
    let out = lhs - out;
    out
}

#[test]
fn test_simple_add() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Vec4(Elem::Float));
    let rhs = context.create_local(Item::Vec4(Elem::Float));

    kernel_expand(&mut context, lhs, rhs);
}
