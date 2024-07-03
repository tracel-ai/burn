use burn_cube::prelude::*;

#[cube]
pub fn while_not<I: Int>(lhs: I) {
    while lhs != I::from_int(0) {
        let _ = lhs % I::from_int(1);
    }
}

#[cube]
pub fn manual_loop_break<I: Int>(lhs: I) {
    loop {
        if lhs != I::from_int(0) {
            break;
        }
        let _ = lhs % I::from_int(1);
    }
}

#[cube]
pub fn loop_with_return<I: Int>(lhs: I) {
    loop {
        if lhs != I::from_int(0) {
            return;
        }
        let _ = lhs % I::from_int(1);
    }
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        ir::{Branch, Elem, Item, Variable},
    };

    type ElemType = I32;

    #[test]
    fn cube_while_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(ElemType::as_elem()));

        while_not_expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref(false));
    }

    #[test]
    fn cube_loop_break_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(ElemType::as_elem()));

        manual_loop_break_expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref(false));
    }

    #[test]
    fn cube_loop_with_return_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(ElemType::as_elem()));

        loop_with_return_expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref(true));
    }

    fn inline_macro_ref(is_return: bool) -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());
        let lhs = context.create_local(item);

        let mut scope = context.into_scope();
        let cond = scope.create_local(Item::new(Elem::Bool));
        let lhs: Variable = lhs.into();
        let rhs = scope.create_local(item);

        cpa!(
            &mut scope,
            loop(|scope| {
                cpa!(scope, cond = lhs != 0);
                cpa!(scope, if(cond).then(|scope|{
                    match is_return {
                        true => scope.register(Branch::Return),
                        false => scope.register(Branch::Break)
                    }
                }));

                cpa!(scope, rhs = lhs % 1i32);
            })
        );

        format!("{:?}", scope.operations)
    }
}
