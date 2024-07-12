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

#[cube]
pub fn elsif<F: Float>(lhs: F) {
    if lhs < F::new(0.) {
        let _ = lhs + F::new(2.);
    } else if lhs > F::new(0.) {
        let _ = lhs + F::new(1.);
    } else {
        let _ = lhs + F::new(0.);
    }
}

mod tests {
    use burn_cube::{
        cpa,
        frontend::{CubeContext, CubePrimitive, F32},
        ir::{Elem, Item, Variable},
    };

    use super::*;

    type ElemType = F32;

    #[test]
    fn cube_if_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(ElemType::as_elem()));

        if_greater::__expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_if());
    }

    #[test]
    fn cube_if_else_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(ElemType::as_elem()));

        if_then_else::__expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_if_else()
        );
    }

    #[test]
    fn cube_elsif_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(ElemType::as_elem()));

        elsif::__expand::<ElemType>(&mut context, lhs);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_elsif());
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

    fn inline_macro_ref_elsif() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());
        let lhs = context.create_local(item);

        let mut scope = context.into_scope();
        let cond1 = scope.create_local(Item::new(Elem::Bool));
        let lhs: Variable = lhs.into();
        let y = scope.create_local(item);
        let cond2 = scope.create_local(Item::new(Elem::Bool));

        cpa!(scope, cond1 = lhs < 0f32);
        cpa!(&mut scope, if(cond1).then(|scope| {
            cpa!(scope, y = lhs + 2.0f32);
        }).else(|mut scope|{
            cpa!(scope, cond2 = lhs > 0f32);
            cpa!(&mut scope, if(cond2).then(|scope| {
                cpa!(scope, y = lhs + 1.0f32);
            }).else(|scope|{
                cpa!(scope, y = lhs + 0.0f32);
            }));
        }));

        format!("{:?}", scope.operations)
    }
}
