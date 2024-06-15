use burn_cube::prelude::*;

mod elsewhere {
    use super::*;

    #[cube]
    pub fn my_func<F: Float>(x: F) -> F {
        x * F::from_int(2)
    }
}

mod here {
    use super::*;

    #[cube]
    pub fn caller<F: Float>(x: F) {
        let _ = x + elsewhere::my_func::<F>(x);
    }

    #[cube]
    pub fn no_call_ref<F: Float>(x: F) {
        let _ = x + x * F::from_int(2);
    }
}

mod tests {
    use super::*;
    use burn_cube::ir::Item;

    type ElemType = F32;

    #[test]
    fn cube_call_equivalent_to_no_call_no_arg_test() {
        let mut caller_context = CubeContext::root();
        let x = caller_context.create_local(Item::new(ElemType::as_elem()));
        here::caller_expand::<ElemType>(&mut caller_context, x);
        let caller_scope = caller_context.into_scope();

        let mut no_call_context = CubeContext::root();
        let x = no_call_context.create_local(Item::new(ElemType::as_elem()));
        here::no_call_ref_expand::<ElemType>(&mut no_call_context, x);
        let no_call_scope = no_call_context.into_scope();

        assert_eq!(
            format!("{:?}", caller_scope.operations),
            format!("{:?}", no_call_scope.operations)
        );
    }
}
