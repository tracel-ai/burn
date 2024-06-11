use burn_cube::prelude::*;

#[cube]
fn mut_assign() {
    let mut x = UInt::new(0);
    x += UInt::new(1);
}

#[cube]
fn mut_assign_input(y: UInt) -> UInt {
    let mut x = y;
    x += UInt::new(1);
    y + UInt::new(2)
}

#[cube]
fn assign_mut_input(mut y: UInt) -> UInt {
    let x = y;
    y += UInt::new(1);
    x + UInt::new(2)
}

#[cube]
fn assign_vectorized(y: UInt) -> UInt {
    let vectorization_factor = Comptime::vectorization(y);
    let x = UInt::vectorized(1, Comptime::get(vectorization_factor));
    x + y
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        ir::{Elem, Item, Variable},
    };

    #[test]
    fn cube_mut_assign_test() {
        let mut context = CubeContext::root();

        mut_assign_expand(&mut context);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_mut_assign()
        );
    }

    #[test]
    fn cube_mut_assign_input_test() {
        let mut context = CubeContext::root();

        let y = context.create_local(Item::new(UInt::as_elem()));

        mut_assign_input_expand(&mut context, y);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_mut_assign_input()
        );
    }

    #[test]
    fn cube_assign_mut_input_test() {
        let mut context = CubeContext::root();

        let y = context.create_local(Item::new(UInt::as_elem()));

        assign_mut_input_expand(&mut context, y);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_assign_mut_input()
        );
    }

    #[test]
    fn cube_assign_vectorized_test() {
        let mut context = CubeContext::root();

        let y = context.create_local(Item::vectorized(UInt::as_elem(), 4));

        assign_vectorized_expand(&mut context, y);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_assign_vectorized()
        );
    }

    fn inline_macro_ref_mut_assign() -> String {
        let context = CubeContext::root();

        let mut scope = context.into_scope();
        let x = scope.create_local(Item::new(Elem::UInt));

        let zero = Variable::ConstantScalar(0., Elem::UInt);
        let one = Variable::ConstantScalar(1., Elem::UInt);
        cpa!(scope, x = zero);
        cpa!(scope, x = x + one);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_mut_assign_input() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(Elem::UInt);
        let y = context.create_local(item);

        let mut scope = context.into_scope();
        let y: Variable = y.into();
        let x = scope.create_local(item);

        let one = Variable::ConstantScalar(1., Elem::UInt);
        let two = Variable::ConstantScalar(2., Elem::UInt);
        cpa!(scope, x = y);
        cpa!(scope, x = x + one);
        cpa!(scope, y = y + two);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_assign_mut_input() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(Elem::UInt);
        let y = context.create_local(item);

        let mut scope = context.into_scope();
        let y: Variable = y.into();
        let x = scope.create_local(item);

        let one = Variable::ConstantScalar(1., Elem::UInt);
        let two = Variable::ConstantScalar(2., Elem::UInt);
        cpa!(scope, x = y);
        cpa!(scope, y = y + one);
        cpa!(scope, x = x + two);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_assign_vectorized() -> String {
        let mut context = CubeContext::root();
        let item = Item::vectorized(Elem::UInt, 4);
        let y = context.create_local(item);

        let mut scope = context.into_scope();
        let y: Variable = y.into();
        let x = scope.create_local(item);

        let zero = Variable::ConstantScalar(0., Elem::UInt);
        let one = Variable::ConstantScalar(1., Elem::UInt);
        let two = Variable::ConstantScalar(2., Elem::UInt);
        let three = Variable::ConstantScalar(3., Elem::UInt);
        cpa!(scope, x[zero] = one);
        cpa!(scope, x[one] = one);
        cpa!(scope, x[two] = one);
        cpa!(scope, x[three] = one);
        cpa!(scope, x = x + y);

        format!("{:?}", scope.operations)
    }
}
