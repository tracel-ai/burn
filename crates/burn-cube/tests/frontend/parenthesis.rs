use burn_cube::prelude::*;

#[cube]
pub fn parenthesis<T: Numeric>(x: T, y: T, z: T) -> T {
    x * (y + z)
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        ir::{Item, Variable},
    };

    type ElemType = F32;

    #[test]
    fn cube_parenthesis_priority_test() {
        let mut context = CubeContext::root();

        let x = context.create_local(Item::new(ElemType::as_elem()));
        let y = context.create_local(Item::new(ElemType::as_elem()));
        let z = context.create_local(Item::new(ElemType::as_elem()));

        parenthesis_expand::<ElemType>(&mut context, x, y, z);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
    }

    fn inline_macro_ref() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local(item);
        let y = context.create_local(item);
        let z = context.create_local(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let y: Variable = y.into();
        let z: Variable = z.into();

        cpa!(scope, y = y + z);
        cpa!(scope, x = x * y);

        format!("{:?}", scope.operations)
    }
}
