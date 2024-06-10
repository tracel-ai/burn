use burn_cube::prelude::*;

#[cube]
pub fn redeclare_same_scope<I: Int>(mut x: I) {
    let i = I::new(1);
    x += i;
    let i = I::new(2);
    x += i;
}

#[cube]
pub fn redeclare_same_scope_other_type<I: Int, F: Float>(mut x: I) -> F {
    let i = I::new(1);
    x += i;
    let i = F::new(2.);
    i
}

#[cube]
pub fn redeclare_different_scope<I: Int>(mut x: I) {
    let y = I::new(1);
    x += y;
    for i in range(0u32, 2u32, Comptime::new(false)) {
        let y = I::new(2);
        x += y;
    }
}

mod tests {
    use burn_cube::{
        cpa,
        ir::{Item, Variable},
    };

    use super::*;

    type ElemType = I32;

    #[test]
    fn cube_redeclare_same_scope_test() {
        let mut context = CubeContext::root();

        let x = context.create_local(Item::new(ElemType::as_elem()));

        redeclare_same_scope_expand::<ElemType>(&mut context, x);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), "".to_string());
    }

    #[test]
    fn cube_redeclare_same_scope_other_type_test() {
        let mut context = CubeContext::root();

        let x = context.create_local(Item::new(ElemType::as_elem()));

        redeclare_same_scope_other_type_expand::<ElemType, F32>(&mut context, x);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), "".to_string());
    }

    #[test]
    fn cube_redeclare_different_scope_test() {
        let mut context = CubeContext::root();

        let x = context.create_local(Item::new(ElemType::as_elem()));

        redeclare_different_scope_expand::<ElemType>(&mut context, x);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_different()
        );
    }

    fn inline_macro_ref_different() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());

        let x = context.create_local(item);
        let end = 2u32;
        let mut scope = context.into_scope();
        let x: Variable = x.into();

        // Kernel
        let y = scope.create_with_value(1, item);
        cpa!(scope, x += y);

        cpa!(
            &mut scope,
            range(0u32, end, false).for_each(|i, scope| {
                let value = Variable::ConstantScalar(2.into(), item.elem());
                cpa!(scope, y = value);
                cpa!(scope, x += y);
            })
        );

        format!("{:?}", scope.operations)
    }
}
