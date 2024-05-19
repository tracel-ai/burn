use burn_cube::{cube, Bool, Cast, Numeric, UInt, F32, I32};

// From float
#[cube]
// #[allow(clippy::useless_conversion)]
pub fn float_to_float(x: F32) {
    let y = x + F32::from_int(2);
    let _ = F32::cast_from(y) + F32::from_int(34);
}

#[cube]
pub fn float_to_int(x: F32) {
    let y = x + F32::from_int(2);
    let _ = I32::cast_from(y) + I32::from_int(34);
}

#[cube]
pub fn float_to_uint(x: F32) {
    let y = x + F32::from_int(2);
    let _ = UInt::cast_from(y) + UInt::from_int(34);
}

#[cube]
pub fn float_to_bool(x: F32) {
    let y = x + F32::from_int(2);
    let _ = Bool::cast_from(y) | Bool::new(true);
}

// From int
#[cube]
pub fn int_to_float(x: I32) {
    let y = x + I32::from_int(2);
    let _ = F32::cast_from(y) + F32::from_int(34);
}

#[cube]
#[allow(clippy::useless_conversion)]
pub fn int_to_int(x: I32) {
    let y = x + I32::from_int(2);
    let _ = I32::cast_from(y) + I32::from_int(34);
}

#[cube]
pub fn int_to_uint(x: I32) {
    let y = x + I32::from_int(2);
    let _ = UInt::cast_from(y) + UInt::from_int(34);
}

#[cube]
pub fn int_to_bool(x: I32) {
    let y = x + I32::from_int(2);
    let _ = Bool::cast_from(y) | Bool::new(true);
}

// // From uint
#[cube]
pub fn uint_to_float(x: UInt) {
    let y = x + UInt::from_int(2);
    let _ = F32::cast_from(y) + F32::from_int(34);
}

#[cube]
pub fn uint_to_int(x: UInt) {
    let y = x + UInt::from_int(2);
    let _ = I32::cast_from(y) + I32::from_int(34);
}

#[cube]
#[allow(clippy::useless_conversion)]
pub fn uint_to_uint(x: UInt) {
    let y = x + UInt::from_int(2);
    let _ = UInt::cast_from(y) + UInt::from_int(34);
}

#[cube]
pub fn uint_to_bool(x: UInt) {
    let y = x + UInt::from_int(2);
    let _ = Bool::cast_from(y) | Bool::new(true);
}

// From bool
#[cube]
pub fn bool_to_float(x: Bool) {
    let y = x & Bool::new(false);
    let _ = F32::cast_from(y) + F32::from_int(34);
}

#[cube]
pub fn bool_to_int(x: Bool) {
    let y = x & Bool::new(false);
    let _ = I32::cast_from(y) + I32::from_int(34);
}

#[cube]
pub fn bool_to_uint(x: Bool) {
    let y = x & Bool::new(false);
    let _ = UInt::cast_from(y) + UInt::from_int(34);
}

#[cube]
#[allow(clippy::useless_conversion)]
pub fn bool_to_bool(x: Bool) {
    let y = x & Bool::new(false);
    let _ = Bool::cast_from(y) | Bool::new(true);
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        dialect::{Elem, Item, Variable},
        CubeContext, PrimitiveVariable,
    };

    macro_rules! cast_test {
        ($name:ident, $module:ident, $from:expr, $to:expr) => {
            #[test]
            fn $name() {
                let mut context = CubeContext::root();

                let x = context.create_local($from);

                $module(&mut context, x);
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

                $module(&mut context, x);
                let scope = context.into_scope();

                assert_eq!(
                    format!("{:?}", scope.operations),
                    inline_macro_ref_identity($ty)
                );
            }
        };
    }

    cast_test!(
        cube_float_to_float_test,
        float_to_float_expand,
        Item::Scalar(F32::into_elem())
    );

    cast_test!(
        cube_float_to_int_test,
        float_to_int_expand,
        Item::Scalar(F32::into_elem()),
        Item::Scalar(I32::into_elem())
    );

    cast_test!(
        cube_float_to_uint_test,
        float_to_uint_expand,
        Item::Scalar(F32::into_elem()),
        Item::Scalar(Elem::UInt)
    );

    cast_test!(
        cube_float_to_bool_test,
        float_to_bool_expand,
        Item::Scalar(F32::into_elem()),
        Item::Scalar(Elem::Bool)
    );

    cast_test!(
        cube_int_to_float_test,
        int_to_float_expand,
        Item::Scalar(I32::into_elem()),
        Item::Scalar(F32::into_elem())
    );

    cast_test!(
        cube_int_to_int_test,
        int_to_int_expand,
        Item::Scalar(I32::into_elem())
    );

    cast_test!(
        cube_int_to_uint_test,
        int_to_uint_expand,
        Item::Scalar(I32::into_elem()),
        Item::Scalar(Elem::UInt)
    );

    cast_test!(
        cube_int_to_bool_test,
        int_to_bool_expand,
        Item::Scalar(I32::into_elem()),
        Item::Scalar(Elem::Bool)
    );

    cast_test!(
        cube_uint_to_float_test,
        uint_to_float_expand,
        Item::Scalar(Elem::UInt),
        Item::Scalar(F32::into_elem())
    );

    cast_test!(
        cube_uint_to_int_test,
        uint_to_int_expand,
        Item::Scalar(Elem::UInt),
        Item::Scalar(I32::into_elem())
    );

    cast_test!(
        cube_uint_to_uint_test,
        uint_to_uint_expand,
        Item::Scalar(Elem::UInt)
    );

    cast_test!(
        cube_uint_to_bool_test,
        uint_to_bool_expand,
        Item::Scalar(Elem::UInt),
        Item::Scalar(Elem::Bool)
    );

    cast_test!(
        cube_bool_to_float_test,
        bool_to_float_expand,
        Item::Scalar(Elem::Bool),
        Item::Scalar(F32::into_elem())
    );

    cast_test!(
        cube_bool_to_int_test,
        bool_to_int_expand,
        Item::Scalar(Elem::Bool),
        Item::Scalar(I32::into_elem())
    );

    cast_test!(
        cube_bool_to_uint_test,
        bool_to_uint_expand,
        Item::Scalar(Elem::Bool),
        Item::Scalar(Elem::UInt)
    );

    cast_test!(
        cube_bool_to_bool_test,
        bool_to_bool_expand,
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
            Elem::Float(_) => cpa!(scope, y = x + 2f32),
            Elem::Int(_) => cpa!(scope, y = x + 2i32),
            Elem::UInt => cpa!(scope, y = x + 2u32),
            Elem::Bool => cpa!(scope, y = x && false),
        }

        cpa!(scope, y_casted = cast(y));

        match to_item.elem() {
            Elem::Float(_) => cpa!(scope, z = y_casted + 34f32),
            Elem::Int(_) => cpa!(scope, z = y_casted + 34i32),
            Elem::UInt => cpa!(scope, z = y_casted + 34u32),
            Elem::Bool => cpa!(scope, z = y_casted || true),
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
            Elem::Float(_) => cpa!(scope, y = x + 2f32),
            Elem::Int(_) => cpa!(scope, y = x + 2i32),
            Elem::UInt => cpa!(scope, y = x + 2u32),
            Elem::Bool => cpa!(scope, y = x && false),
        }

        cpa!(scope, x = cast(y));

        match item.elem() {
            Elem::Float(_) => cpa!(scope, y = x + 34f32),
            Elem::Int(_) => cpa!(scope, y = x + 34i32),
            Elem::UInt => cpa!(scope, y = x + 34u32),
            Elem::Bool => cpa!(scope, y = x || true),
        }

        format!("{:?}", scope.operations)
    }
}
