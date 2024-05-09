// use burn_cube::{cube, CubeContext, FloatX, Float, F32_, F64_};
// use burn_jit::gpu;
// use burn_jit::gpu::FloatKind;
// use burn_jit::gpu::{Elem, Item};

// #[cube]
// pub fn cast_float_kind<F1: Float, F2: Float>(input: FloatX<F1>) {
//     let x = input + float_new::<F1>(5.9f32);
//     let y = to_float::<FloatX<F1>, F2>(x);
//     let _ = y + float_new::<F2>(2.3f32);
// }

// #[test]
// fn cube_cast_kind_test() {
//     let mut context = CubeContext::root();
//     let item = Item::Scalar(Elem::Float(FloatKind::F64));

//     let input = context.create_local(item);

//     // F16 not testable with the gpu macro, but should work the same
//     cast_float_kind::expand::<F64_, F32_>(&mut context, input);
//     let scope = context.into_scope();

//     assert_eq!(format!("{:?}", scope.operations), gpu_macro_ref());
// }

// fn gpu_macro_ref() -> String {
//     let mut context = CubeContext::root();
//     let float_64 = Item::Scalar(Elem::Float(FloatKind::F64));
//     let float_32 = Item::Scalar(Elem::Float(FloatKind::F32));
//     let input = context.create_local(float_64);

//     let mut scope = context.into_scope();
//     let x = scope.create_local(float_64);
//     let y = scope.create_local(float_32);
//     let z = scope.create_local(float_32);

//     gpu!(scope, x = input + 5.9f32 as f64);
//     gpu!(scope, y = cast(x));
//     gpu!(scope, z = y + 2.3f32);

//     format!("{:?}", scope.operations)
// }
