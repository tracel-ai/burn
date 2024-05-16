use burn_cube::{cube, Cast, Float, Int, Numeric};

#[cube]
pub fn cast_float_kind<F1: Float, F2: Float>(input: F1) {
    let x = input + F1::from_primitive(5.9);
    let y = F2::cast_from(x);
    let _ = y + F2::from_primitive(2.3);
}

#[cube]
pub fn cast_int_kind<I1: Int, I2: Int>(input: I1) {
    let x = input + I1::from_primitive(5);
    let y = I2::cast_from(x);
    let _ = y + I2::from_primitive(2);
}

#[cube]
pub fn cast_numeric_to_kind<T: Numeric, I: Int>(input: T) {
    let x = input + T::lit(5);
    let y = I::cast_from(x);
    let _ = y + I::lit(2);
}

#[cube]
pub fn cast_int_to_numeric<I: Int, T: Numeric>(input: I) {
    let x = input + I::lit(5);
    let y = T::cast_from(x);
    let _ = y + T::lit(2);
}

mod tests {
    use super::*;
    use burn_cube::{cpa, dialect::Item, CubeContext, PrimitiveVariable, F32, F64, I32, I64};

    #[test]
    fn cube_cast_float_kind_test() {
        let mut context = CubeContext::root();
        let item = Item::Scalar(F64::into_elem());

        let input = context.create_local(item);

        cast_float_kind_expand::<F64, F32>(&mut context, input);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_float());
    }

    #[test]
    fn cube_cast_int_kind_test() {
        let mut context = CubeContext::root();
        let item = Item::Scalar(I32::into_elem());

        let input = context.create_local(item);

        cast_int_kind_expand::<I32, I64>(&mut context, input);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
    }

    #[test]
    fn cube_cast_numeric_kind_test() {
        let mut context = CubeContext::root();
        let item = Item::Scalar(I32::into_elem());

        let input = context.create_local(item);

        cast_numeric_to_kind_expand::<I32, I64>(&mut context, input);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
    }

    #[test]
    fn cube_cast_kind_numeric_test() {
        let mut context = CubeContext::root();
        let item = Item::Scalar(I32::into_elem());

        let input = context.create_local(item);

        cast_int_to_numeric_expand::<I32, I64>(&mut context, input);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
    }

    fn inline_macro_ref_float() -> String {
        let mut context = CubeContext::root();
        let float_64 = Item::Scalar(F64::into_elem());
        let float_32 = Item::Scalar(F32::into_elem());
        let input = context.create_local(float_64);

        let mut scope = context.into_scope();
        let x = scope.create_local(float_64);
        let y = scope.create_local(float_32);
        let z = scope.create_local(float_32);

        cpa!(scope, x = input + 5.9f32 as f64);
        cpa!(scope, y = cast(x));
        cpa!(scope, z = y + 2.3f32);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_int() -> String {
        let mut context = CubeContext::root();
        let int_32 = Item::Scalar(I32::into_elem());
        let int_64 = Item::Scalar(I64::into_elem());
        let input = context.create_local(int_32);

        let mut scope = context.into_scope();
        let x = scope.create_local(int_32);
        let y = scope.create_local(int_64);
        let z = scope.create_local(int_64);

        cpa!(scope, x = input + 5i32);
        cpa!(scope, y = cast(x));
        cpa!(scope, z = y + 2i64);

        format!("{:?}", scope.operations)
    }
}
