use burn_cube::{cube, Bool, CubeContext, Float, Int, UInt, F32, I32};
use burn_jit::{
    cube_inline,
    gpu::{Elem, Item, Variable},
};

macro_rules! cast_test {
    ($name:ident, $module:ident, $from:expr, $to:expr) => {
        #[test]
        fn $name() {
            let mut context = CubeContext::root();

            let x = context.create_local($from);

            $module::expand(&mut context, x);
            let scope = context.into_scope();

            assert_eq!(
                format!("{:?}", scope.operations),
                inline_macro_ref_cast($from, $to)
            );
        }
    };

    ($name:ident, $module:ident, $ty:expr) => {
        #[test]
        fn $name() {
            let mut context = CubeContext::root();

            let x = context.create_local($ty);

            $module::expand(&mut context, x);
            let scope = context.into_scope();

            assert_eq!(
                format!("{:?}", scope.operations),
                inline_macro_ref_identity($ty)
            );
        }
    };
}

// From float
#[cube]
pub fn float_to_float(x: F32) {
    let y = x + float_new::<F32>(2.0);
    let _ = to_float::<F32, F32>(y) + float_new::<F32>(34.0);
}

#[cube]
pub fn float_to_int(x: F32) {
    let y = x + float_new::<F32>(2.0);
    let _ = to_int::<F32, I32>(y) + int_new::<I32>(34);
}

#[cube]
pub fn float_to_uint(x: F32) {
    let y = x + float_new::<F32>(2.0);
    let _ = to_uint(y) + uint_new(34u32);
}

#[cube]
pub fn float_to_bool(x: F32) {
    let y = x + float_new::<F32>(2.0);
    let _ = to_bool(y) | bool_new(true);
}

cast_test!(
    cube_float_to_float_test,
    float_to_float,
    Item::Scalar(Elem::Float(F32::into_kind()))
);

cast_test!(
    cube_float_to_int_test,
    float_to_int,
    Item::Scalar(Elem::Float(F32::into_kind())),
    Item::Scalar(Elem::Int(I32::into_kind()))
);

cast_test!(
    cube_float_to_uint_test,
    float_to_uint,
    Item::Scalar(Elem::Float(F32::into_kind())),
    Item::Scalar(Elem::UInt)
);

cast_test!(
    cube_float_to_bool_test,
    float_to_bool,
    Item::Scalar(Elem::Float(F32::into_kind())),
    Item::Scalar(Elem::Bool)
);

// // From int
#[cube]
pub fn int_to_float(x: I32) {
    let y = x + int_new::<I32>(2);
    let _ = to_float::<I32, F32>(y) + float_new::<F32>(34.0);
}

#[cube]
pub fn int_to_int(x: I32) {
    let y = x + int_new::<I32>(2);
    let _ = to_int::<I32, I32>(y) + int_new::<I32>(34);
}

#[cube]
pub fn int_to_uint(x: I32) {
    let y = x + int_new::<I32>(2);
    let _ = to_uint(y) + uint_new(34u32);
}

#[cube]
pub fn int_to_bool(x: I32) {
    let y = x + int_new::<I32>(2);
    let _ = to_bool(y) | bool_new(true);
}

cast_test!(
    cube_int_to_float_test,
    int_to_float,
    Item::Scalar(Elem::Int(I32::into_kind())),
    Item::Scalar(Elem::Float(F32::into_kind()))
);

cast_test!(
    cube_int_to_int_test,
    int_to_int,
    Item::Scalar(Elem::Int(I32::into_kind()))
);

cast_test!(
    cube_int_to_uint_test,
    int_to_uint,
    Item::Scalar(Elem::Int(I32::into_kind())),
    Item::Scalar(Elem::UInt)
);

cast_test!(
    cube_int_to_bool_test,
    int_to_bool,
    Item::Scalar(Elem::Int(I32::into_kind())),
    Item::Scalar(Elem::Bool)
);

// // From uint
#[cube]
pub fn uint_to_float(x: UInt) {
    let y = x + uint_new(2u32);
    let _ = to_float::<UInt, F32>(y) + float_new::<F32>(34.0);
}

#[cube]
pub fn uint_to_int(x: UInt) {
    let y = x + uint_new(2u32);
    let _ = to_int::<UInt, I32>(y) + int_new::<I32>(34);
}

#[cube]
pub fn uint_to_uint(x: UInt) {
    let y = x + uint_new(2u32);
    let _ = to_uint(y) + uint_new(34u32);
}

#[cube]
pub fn uint_to_bool(x: UInt) {
    let y = x + uint_new(2u32);
    let _ = to_bool(y) | bool_new(true);
}

cast_test!(
    cube_uint_to_float_test,
    uint_to_float,
    Item::Scalar(Elem::UInt),
    Item::Scalar(Elem::Float(F32::into_kind()))
);

cast_test!(
    cube_uint_to_int_test,
    uint_to_int,
    Item::Scalar(Elem::UInt),
    Item::Scalar(Elem::Int(I32::into_kind()))
);

cast_test!(
    cube_uint_to_uint_test,
    uint_to_uint,
    Item::Scalar(Elem::UInt)
);

cast_test!(
    cube_uint_to_bool_test,
    uint_to_bool,
    Item::Scalar(Elem::UInt),
    Item::Scalar(Elem::Bool)
);

// From bool
#[cube]
pub fn bool_to_float(x: Bool) {
    let y = x & bool_new(false);
    let _ = to_float::<Bool, F32>(y) + float_new::<F32>(34.0);
}

#[cube]
pub fn bool_to_int(x: Bool) {
    let y = x & bool_new(false);
    let _ = to_int::<Bool, I32>(y) + int_new::<I32>(34);
}

#[cube]
pub fn bool_to_uint(x: Bool) {
    let y = x & bool_new(false);
    let _ = to_uint(y) + uint_new(34u32);
}

#[cube]
pub fn bool_to_bool(x: Bool) {
    let y = x & bool_new(false);
    let _ = to_bool(y) | bool_new(true);
}

cast_test!(
    cube_bool_to_float_test,
    bool_to_float,
    Item::Scalar(Elem::Bool),
    Item::Scalar(Elem::Float(F32::into_kind()))
);

cast_test!(
    cube_bool_to_int_test,
    bool_to_int,
    Item::Scalar(Elem::Bool),
    Item::Scalar(Elem::Int(I32::into_kind()))
);

cast_test!(
    cube_bool_to_uint_test,
    bool_to_uint,
    Item::Scalar(Elem::Bool),
    Item::Scalar(Elem::UInt)
);

cast_test!(
    cube_bool_to_bool_test,
    bool_to_bool,
    Item::Scalar(Elem::Bool)
);

fn inline_macro_ref_cast(from_item: Item, to_item: Item) -> String {
    let mut context = CubeContext::root();
    let x = context.create_local(from_item);

    let mut scope = context.into_scope();
    let x: Variable = x.into();
    let y = scope.create_local(from_item);
    let y_casted = scope.create_local(to_item);
    let z = scope.create_local(to_item);

    match from_item.elem() {
        Elem::Float(_) => cube_inline!(scope, y = x + 2f32),
        Elem::Int(_) => cube_inline!(scope, y = x + 2i32),
        Elem::UInt => cube_inline!(scope, y = x + 2u32),
        Elem::Bool => cube_inline!(scope, y = x && false),
    }

    cube_inline!(scope, y_casted = cast(y));

    match to_item.elem() {
        Elem::Float(_) => cube_inline!(scope, z = y_casted + 34f32),
        Elem::Int(_) => cube_inline!(scope, z = y_casted + 34i32),
        Elem::UInt => cube_inline!(scope, z = y_casted + 34u32),
        Elem::Bool => cube_inline!(scope, z = y_casted || true),
    }

    format!("{:?}", scope.operations)
}

fn inline_macro_ref_identity(item: Item) -> String {
    // When staying with the same type variables are automatically reused in cube
    let mut context = CubeContext::root();
    let x = context.create_local(item);

    let mut scope = context.into_scope();
    let x: Variable = x.into();
    let y = scope.create_local(item);

    match item.elem() {
        Elem::Float(_) => cube_inline!(scope, y = x + 2f32),
        Elem::Int(_) => cube_inline!(scope, y = x + 2i32),
        Elem::UInt => cube_inline!(scope, y = x + 2u32),
        Elem::Bool => cube_inline!(scope, y = x && false),
    }

    cube_inline!(scope, x = cast(y));

    match item.elem() {
        Elem::Float(_) => cube_inline!(scope, y = x + 34f32),
        Elem::Int(_) => cube_inline!(scope, y = x + 34i32),
        Elem::UInt => cube_inline!(scope, y = x + 34u32),
        Elem::Bool => cube_inline!(scope, y = x || true),
    }

    format!("{:?}", scope.operations)
}
