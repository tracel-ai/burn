use burn_cube::{cube, Numeric};

#[cube]
pub fn if_greater<T: Numeric>(lhs: T) {
    if lhs > T::from_int(0) {
        let _ = lhs + T::from_int(4);
    }
}

mod tests {
    use burn_cube::{
        cpa,
        dialect::{Elem, Item, Variable},
        CubeContext, PrimitiveVariable, F32,
    };

    use crate::if_greater_expand;

    type ElemType = F32;

    #[test]
    fn cube_if_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::Scalar(ElemType::into_elem()));

        if_greater_expand::<ElemType>(&mut context, lhs);
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
        let y = scope.create_local(item);

        cpa!(scope, cond = lhs > 0f32);
        cpa!(&mut scope, if(cond).then(|scope| {
            cpa!(scope, y = lhs + 4.0f32);
        }));

        format!("{:?}", scope.operations)
    }
}
