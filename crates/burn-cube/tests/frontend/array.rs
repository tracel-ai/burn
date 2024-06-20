use burn_cube::prelude::*;

#[cube]
fn array_assign_simple(mut array: Array<UInt>) {
    array[UInt::new(1)] += UInt::new(1);
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        ir::{Elem, Item, Variable},
    };

    #[test]
    fn array_assign_simple() {
        let mut context = CubeContext::root();
        let array = context.input(0, Item::new(Elem::UInt));

        array_assign_simple_expand(&mut context, array);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_array_assign_simple()
        );
    }

    fn inline_macro_array_assign_simple() -> String {
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
}
