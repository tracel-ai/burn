use burn_cube::prelude::*;

#[cube]
fn array_add_assign_simple(mut array: Array<UInt>) {
    array[UInt::new(1)] += UInt::new(1);
}

#[cube]
fn array_add_assign_expr(mut array: Array<UInt>) {
    array[UInt::new(1) + UInt::new(5)] += UInt::new(1);
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        ir::{Elem, Item, Variable},
    };

    #[test]
    fn array_add_assign() {
        let mut context = CubeContext::root();
        let array = context.input(0, Item::new(Elem::UInt));

        array_add_assign_simple_expand(&mut context, array);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_array_add_assign_simple()
        );
    }

    #[test]
    fn array_add_assign_expr() {
        let mut context = CubeContext::root();
        let array = context.input(0, Item::new(Elem::UInt));

        array_add_assign_expr_expand(&mut context, array);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_array_add_assign_expr()
        );
    }

    fn inline_macro_array_add_assign_simple() -> String {
        let context = CubeContext::root();

        let mut scope = context.into_scope();
        let local = scope.create_local(Item::new(Elem::UInt));

        let array = Variable::GlobalInputArray(0, Item::new(Elem::UInt));
        let index = Variable::ConstantScalar(1., Elem::UInt);
        let value = Variable::ConstantScalar(1., Elem::UInt);

        cpa!(scope, local = array[index]);
        cpa!(scope, local += value);
        cpa!(scope, array[index] = local);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_array_add_assign_expr() -> String {
        let context = CubeContext::root();

        let mut scope = context.into_scope();
        let index = scope.create_local(Item::new(Elem::UInt));
        let local = scope.create_local(Item::new(Elem::UInt));

        let array = Variable::GlobalInputArray(0, Item::new(Elem::UInt));
        let const1 = Variable::ConstantScalar(1., Elem::UInt);
        let const2 = Variable::ConstantScalar(5., Elem::UInt);
        let value = Variable::ConstantScalar(1., Elem::UInt);

        cpa!(scope, index = const1 + const2);
        cpa!(scope, local = array[index]);
        cpa!(scope, local += value);
        cpa!(scope, array[index] = local);

        format!("{:?}", scope.operations)
    }
}
