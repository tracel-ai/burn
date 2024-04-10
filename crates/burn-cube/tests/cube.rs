use burn_cube::{cube, range, range_expand, CubeContext, Float, UInt};
use burn_jit::gpu::{Elem, Item};

#[cube]
pub fn kernel(lhs: Float, rhs: Float, end: UInt) -> Float {
    let mut out = lhs.clone() + rhs.clone();

    for i in range(0, end, false) {
        let temp = out.clone() * rhs.clone();
        out = kernel_inner(out.clone(), temp);
    }

    out
}

#[cube]
pub fn kernel_inner(lhs: Float, rhs: Float) -> Float {
    lhs + rhs
}

#[test]
fn test_simple_add() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Vec4(Elem::Float));
    let rhs = context.create_local(Item::Vec4(Elem::Float));
    let end = context.create_local(Item::Scalar(Elem::UInt));

    kernel_expand(&mut context, lhs, rhs, end);
}
