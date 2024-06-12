use burn_cube::{
    cube,
    frontend::{Cast, Numeric, UInt, F32, I32},
};

// From float
#[cube]
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
#[allow(clippy::overly_complex_bool_expr)]
pub fn float_to_bool(x: F32) {
    let y = x + F32::from_int(2);
    let _ = bool::cast_from(y) || true;
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
#[allow(clippy::overly_complex_bool_expr)]
pub fn int_to_bool(x: I32) {
    let y = x + I32::from_int(2);
    let _ = bool::cast_from(y) || true;
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
#[allow(clippy::overly_complex_bool_expr)]
pub fn uint_to_bool(x: UInt) {
    let y = x + UInt::from_int(2);
    let _ = bool::cast_from(y) || true;
}

// From bool
#[cube]
#[allow(clippy::overly_complex_bool_expr)]
pub fn bool_to_float(x: bool) {
    let y = x && false;
    let _ = F32::cast_from(y) + F32::from_int(34);
}

#[cube]
#[allow(clippy::overly_complex_bool_expr)]
pub fn bool_to_int(x: bool) {
    let y = x && false;
    let _ = I32::cast_from(y) + I32::from_int(34);
}

#[cube]
#[allow(clippy::overly_complex_bool_expr)]
pub fn bool_to_uint(x: bool) {
    let y = x && false;
    let _ = UInt::cast_from(y) + UInt::from_int(34);
}

#[cube]
#[allow(clippy::overly_complex_bool_expr)]
#[allow(clippy::useless_conversion)]
pub fn bool_to_bool(x: bool) {
    let y = x && false;
    let _ = bool::cast_from(y) || true;
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        frontend::{CubeContext, CubeElem},
        ir::{Elem, Item, Variable},
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
    }

    cast_test!(
        cube_float_to_float_test,
        float_to_float_expand,
        Item::new(F32::as_elem()),
        Item::new(F32::as_elem())
    );

    cast_test!(
        cube_float_to_int_test,
        float_to_int_expand,
        Item::new(F32::as_elem()),
        Item::new(I32::as_elem())
    );

    cast_test!(
        cube_float_to_uint_test,
        float_to_uint_expand,
        Item::new(F32::as_elem()),
        Item::new(Elem::UInt)
    );

    cast_test!(
        cube_float_to_bool_test,
        float_to_bool_expand,
        Item::new(F32::as_elem()),
        Item::new(Elem::Bool)
    );

    cast_test!(
        cube_int_to_float_test,
        int_to_float_expand,
        Item::new(I32::as_elem()),
        Item::new(F32::as_elem())
    );

    cast_test!(
        cube_int_to_int_test,
        int_to_int_expand,
        Item::new(I32::as_elem()),
        Item::new(I32::as_elem())
    );

    cast_test!(
        cube_int_to_uint_test,
        int_to_uint_expand,
        Item::new(I32::as_elem()),
        Item::new(Elem::UInt)
    );

    cast_test!(
        cube_int_to_bool_test,
        int_to_bool_expand,
        Item::new(I32::as_elem()),
        Item::new(Elem::Bool)
    );

    cast_test!(
        cube_uint_to_float_test,
        uint_to_float_expand,
        Item::new(Elem::UInt),
        Item::new(F32::as_elem())
    );

    cast_test!(
        cube_uint_to_int_test,
        uint_to_int_expand,
        Item::new(Elem::UInt),
        Item::new(I32::as_elem())
    );

    cast_test!(
        cube_uint_to_uint_test,
        uint_to_uint_expand,
        Item::new(Elem::UInt),
        Item::new(Elem::UInt)
    );

    cast_test!(
        cube_uint_to_bool_test,
        uint_to_bool_expand,
        Item::new(Elem::UInt),
        Item::new(Elem::Bool)
    );

    cast_test!(
        cube_bool_to_float_test,
        bool_to_float_expand,
        Item::new(Elem::Bool),
        Item::new(F32::as_elem())
    );

    cast_test!(
        cube_bool_to_int_test,
        bool_to_int_expand,
        Item::new(Elem::Bool),
        Item::new(I32::as_elem())
    );

    cast_test!(
        cube_bool_to_uint_test,
        bool_to_uint_expand,
        Item::new(Elem::Bool),
        Item::new(Elem::UInt)
    );

    cast_test!(
        cube_bool_to_bool_test,
        bool_to_bool_expand,
        Item::new(Elem::Bool),
        Item::new(Elem::Bool)
    );

    fn inline_macro_ref_cast(from_item: Item, to_item: Item) -> String {
        let mut context = CubeContext::root();
        let x = context.create_local(from_item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let y = scope.create_local(to_item);

        match from_item.elem() {
            Elem::Float(_) => cpa!(scope, x = x + 2f32),
            Elem::Int(_) => cpa!(scope, x = x + 2i32),
            Elem::UInt => cpa!(scope, x = x + 2u32),
            Elem::Bool => cpa!(scope, x = x && false),
        }

        cpa!(scope, y = cast(x));

        match to_item.elem() {
            Elem::Float(_) => cpa!(scope, y = y + 34f32),
            Elem::Int(_) => cpa!(scope, y = y + 34i32),
            Elem::UInt => cpa!(scope, y = y + 34u32),
            Elem::Bool => cpa!(scope, y = y || true),
        }

        format!("{:?}", scope.operations)
    }
}
