use burn_cube::{cube, Float};

#[cube]
pub fn literal<F: Float>(lhs: F) {
    let _ = lhs + F::from_int(5);
}

#[cube]
pub fn literal_float_no_decimals<F: Float>(lhs: F) {
    let _ = lhs + F::new(5.);
}

mod tests {
    use super::{literal_expand, literal_float_no_decimals_expand};
    use burn_cube::{cpa, dialect::Item, CubeContext, PrimitiveVariable, F32};

    type ElemType = F32;

    #[test]
    fn cube_literal_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::scalar(ElemType::as_elem()));

        literal_expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
    }

    #[test]
    fn cube_literal_float_no_decimal_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::scalar(ElemType::as_elem()));

        literal_float_no_decimals_expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
    }

    fn inline_macro_ref() -> String {
        let mut context = CubeContext::root();
        let item = Item::scalar(ElemType::as_elem());
        let lhs = context.create_local(item);

        let mut scope = context.into_scope();
        let out = scope.create_local(item);
        cpa!(scope, out = lhs + 5.0f32);

        format!("{:?}", scope.operations)
    }
}
