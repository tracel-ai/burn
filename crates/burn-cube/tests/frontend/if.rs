use burn_cube::prelude::*;

#[cube]
pub fn if_greater<T: Numeric>(lhs: T) {
    if lhs > T::from_int(0) {
        let _ = lhs + T::from_int(4);
    }
}

#[cube]
pub fn if_greater_var<T: Numeric>(lhs: T) {
    let x = lhs > T::from_int(0);
    if x {
        let _ = lhs + T::from_int(4);
    }
}

#[cube]
pub fn if_then_else<F: Float>(lhs: F) {
    if lhs < F::from_int(0) {
        let _ = lhs + F::from_int(4);
    } else {
        let _ = lhs - F::from_int(5);
    }
}

mod tests {
    use burn_cube::{
        cpa,
        frontend::{CubeContext, CubeElem, F32},
        ir::{Elem, Item, Variable},
    };

    use super::*;

    type ElemType = F32;

    #[test]
    fn cube_if_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(ElemType::as_elem()));

        if_greater_expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_if());
    }

    #[test]
    fn cube_if_else_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(ElemType::as_elem()));

        if_then_else_expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_if_else()
        );
    }

    fn inline_macro_ref_if() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());
        let lhs = context.create_local(item);

        let mut scope = context.into_scope();
        let cond = scope.create_local(Item::new(Elem::Bool));
        let lhs: Variable = lhs.into();
        let y = scope.create_local(item);

        cpa!(scope, cond = lhs > 0f32);
        cpa!(&mut scope, if(cond).then(|scope| {
            cpa!(scope, y = lhs + 4.0f32);
        }));

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_if_else() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());
        let lhs = context.create_local(item);

        let mut scope = context.into_scope();
        let cond = scope.create_local(Item::new(Elem::Bool));
        let lhs: Variable = lhs.into();
        let y = scope.create_local(item);

        cpa!(scope, cond = lhs < 0f32);
        cpa!(&mut scope, if(cond).then(|scope| {
            cpa!(scope, y = lhs + 4.0f32);
        }).else(|scope|{
            cpa!(scope, y = lhs - 5.0f32);
        }));

        format!("{:?}", scope.operations)
    }
}
