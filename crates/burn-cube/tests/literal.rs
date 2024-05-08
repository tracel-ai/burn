use burn_cube::{cube, CubeContext, Float, F32_};
use burn_jit::gpu;
use burn_jit::gpu::FloatKind;
use burn_jit::gpu::{Elem, Item};

#[cube]
pub fn literal(lhs: Float<F32_>) {
    let _ = lhs + float_new::<F32_>(5.9);
}

#[test]
fn cube_literal_test() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Float(FloatKind::F32)));

    literal::expand(&mut context, lhs);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref());
}

fn gpu_macro_ref() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(Elem::Float(FloatKind::F32));
    let lhs = context.create_local(item);

    let mut scope = context.into_scope();
    let out = scope.create_local(item);
    gpu!(scope, out = lhs + 5.9f32);

    format!("{:?}", scope.operations)
}
