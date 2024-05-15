use burn_cube::{cube, Int};

#[cube]
pub fn while_not<I: Int>(lhs: I) {
    while lhs != I::lit(0) {
        let _ = lhs % I::lit(1);
    }
}

#[cube]
pub fn manual_loop_break<I: Int>(lhs: I) {
    loop {
        if lhs != I::lit(0) {
            break;
        }
        let _ = lhs % I::lit(1);
    }
}

mod tests {
    use burn_cube::{
        cpa,
        dialect::{Branch, Elem, Item, Variable},
        CubeContext, PrimitiveVariable, I32,
    };

    use crate::{manual_loop_break_expand, while_not_expand};

    type ElemType = I32;

    #[test]
    fn cube_while_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::Scalar(ElemType::into_elem()));

        while_not_expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
    }

    #[test]
    fn cube_loop_break_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::Scalar(ElemType::into_elem()));

        manual_loop_break_expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
    }

    fn inline_macro_ref() -> String {
        let mut context = CubeContext::root();
        let item = Item::Scalar(ElemType::into_elem());
        let lhs = context.create_local(item);

        let mut scope = context.into_scope();
        let cond = scope.create_local(Item::Scalar(Elem::Bool));
        let lhs: Variable = lhs.into();
        let rhs = scope.create_local(item);

        cpa!(
            &mut scope,
            loop(|scope| {
                cpa!(scope, cond = lhs != 0);
                cpa!(scope, if(cond).then(|scope|{
                    scope.register(Branch::Break);
                }));

                cpa!(scope, rhs = lhs % 1i32);
            })
        );

        format!("{:?}", scope.operations)
    }
}
