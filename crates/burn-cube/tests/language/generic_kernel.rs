use burn_cube::{cube, Numeric};

#[cube]
pub fn generic_kernel<T: Numeric>(lhs: T) {
    let _ = lhs + T::from_int(5);
}

mod tests {
    use burn_cube::{cpa, dialect::Item, CubeContext, PrimitiveVariable, F32, I32};

    use super::*;

    #[test]
    fn cube_generic_float_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::scalar(F32::as_elem()));

        generic_kernel_expand::<F32>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_float());
    }

    #[test]
    fn cube_generic_int_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::scalar(I32::as_elem()));

        generic_kernel_expand::<I32>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
    }

    fn inline_macro_ref_float() -> String {
        let mut context = CubeContext::root();
        let item = Item::scalar(F32::as_elem());
        let lhs = context.create_local(item);

        let mut scope = context.into_scope();
        let out = scope.create_local(item);
        cpa!(scope, out = lhs + 5.0f32);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_int() -> String {
        let mut context = CubeContext::root();
        let item = Item::scalar(I32::as_elem());
        let lhs = context.create_local(item);

        let mut scope = context.into_scope();
        let out = scope.create_local(item);
        cpa!(scope, out = lhs + 5);

        format!("{:?}", scope.operations)
    }
}
