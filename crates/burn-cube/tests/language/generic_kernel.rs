use burn_cube::{cube, frontend::Numeric};

#[cube]
pub fn generic_kernel<T: Numeric>(lhs: T) {
    let _ = lhs + T::from_int(5);
}

mod tests {
    use burn_cube::{
        cpa,
        frontend::{CubeContext, CubeElem, F32, I32},
        ir::{Item, Variable},
    };

    use super::*;

    #[test]
    fn cube_generic_float_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(F32::as_elem()));

        generic_kernel_expand::<F32>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_float());
    }

    #[test]
    fn cube_generic_int_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(I32::as_elem()));

        generic_kernel_expand::<I32>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
    }

    fn inline_macro_ref_float() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(F32::as_elem());
        let var = context.create_local(item);

        let mut scope = context.into_scope();
        let var: Variable = var.into();
        cpa!(scope, var = var + 5.0f32);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_int() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(I32::as_elem());
        let var = context.create_local(item);

        let mut scope = context.into_scope();
        let var: Variable = var.into();
        cpa!(scope, var = var + 5);

        format!("{:?}", scope.operations)
    }
}
