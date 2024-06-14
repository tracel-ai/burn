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
    i + i
}

#[cube]
pub fn redeclare_different_scope<I: Int>(mut x: I) {
    let y = I::new(1);
    x += y;
    for _ in range(0u32, 2u32, Comptime::new(false)) {
        let y = I::new(2);
        x += y;
    }
}

#[cube]
pub fn redeclare_two_for_loops(mut x: UInt) {
    for i in range(0u32, 2u32, Comptime::new(false)) {
        x += i;
    }
    for i in range(0u32, 2u32, Comptime::new(false)) {
        x += i;
        x += i;
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

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_same_scope()
        );
    }

    #[test]
    fn cube_redeclare_same_scope_other_type_test() {
        let mut context = CubeContext::root();

        let x = context.create_local(Item::new(ElemType::as_elem()));

        redeclare_same_scope_other_type_expand::<ElemType, F32>(&mut context, x);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_same_scope_other_type()
        );
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

    #[test]
    fn cube_redeclare_two_for_loops_test() {
        let mut context = CubeContext::root();

        let x = context.create_local(Item::new(UInt::as_elem()));

        redeclare_two_for_loops_expand(&mut context, x);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_two_for_loops()
        );
    }

    fn inline_macro_ref_same_scope() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());

        let x = context.create_local(item);
        let mut scope = context.into_scope();
        let x: Variable = x.into();

        let i = scope.create_with_value(1, item);
        cpa!(scope, x += i);
        let value = Variable::ConstantScalar(2., item.elem());
        cpa!(scope, i = value);
        cpa!(scope, x += i);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_same_scope_other_type() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());

        let x = context.create_local(item);
        let mut scope = context.into_scope();
        let x: Variable = x.into();

        let i = scope.create_with_value(1, item);
        cpa!(scope, x += i);
        let i = scope.create_with_value(2, Item::new(F32::as_elem()));
        let y = scope.create_local(Item::new(F32::as_elem()));
        cpa!(scope, y = i + i);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_different() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());

        let x = context.create_local(item);
        let end = 2u32;
        let mut scope = context.into_scope();
        let x: Variable = x.into();

        let y = scope.create_with_value(1, item);
        cpa!(scope, x += y);

        cpa!(
            &mut scope,
            range(0u32, end, false).for_each(|_, scope| {
                let value = Variable::ConstantScalar(2.into(), item.elem());
                cpa!(scope, y = value);
                cpa!(scope, x += y);
            })
        );

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_two_for_loops() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(UInt::as_elem());

        let x = context.create_local(item);
        let end = 2u32;
        let mut scope = context.into_scope();
        let x: Variable = x.into();

        cpa!(
            &mut scope,
            range(0u32, end, false).for_each(|i, scope| {
                cpa!(scope, x += i);
            })
        );

        cpa!(
            &mut scope,
            range(0u32, end, false).for_each(|i, scope| {
                cpa!(scope, x += i);
                cpa!(scope, x += i);
            })
        );

        format!("{:?}", scope.operations)
    }
}
