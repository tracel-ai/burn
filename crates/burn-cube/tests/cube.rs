use burn_cube::{cube, CodegenContext, Float};
use burn_jit::gpu::{Elem, Item, Scope};

#[cube]
pub fn kernel(lhs: Float, rhs: Float) -> Float {
    let mut output = lhs;

    for i in 0..10 {
        output = lhs + rhs
    }

    output
}

#[test]
fn test_simple_add() {
    let mut scope = Scope::root();
    let mut context = CodegenContext {
        scope: &mut scope,
        pool: Default::default(),
    };

    let lhs = context.crate_float(Item::Vec4(Elem::Float));
    let rhs = context.crate_float(Item::Vec4(Elem::Float));

    kernel_expand(&mut context, lhs, rhs);
}
