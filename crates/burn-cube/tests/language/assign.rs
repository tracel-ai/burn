use burn_cube::prelude::*;

#[cube]
fn mut_assign() {
    let mut x = UInt::new(0);
    x += UInt::new(1);
}

#[cube]
fn mut_assign_input(y: UInt) {
    let mut x = y;
    x += UInt::new(1);
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        ir::{Elem, Item, Variable},
    };

    #[test]
    fn cube_a() {
        let mut context = CubeContext::root();

        mut_assign_expand(&mut context);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
    }

    fn inline_macro_ref() -> String {
        let context = CubeContext::root();

        let mut scope = context.into_scope();
        let x = scope.create_local(Item::new(Elem::UInt));

        let zero = Variable::ConstantScalar(0., Elem::UInt);
        let one = Variable::ConstantScalar(1., Elem::UInt);
        cpa!(scope, x = zero);
        cpa!(scope, x = x + one);

        format!("{:?}", scope.operations)
    }
}
