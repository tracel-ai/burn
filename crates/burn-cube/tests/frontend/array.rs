use burn_cube::prelude::*;

#[cube]
pub fn array_read_write<T: Numeric>(array_size: Comptime<u32>) {
    let mut array = Array::<T>::new(array_size);
    array[0] = T::from_int(3);
    let _ = array[0];
}

#[cube]
pub fn array_to_vectorized_variable<T: Numeric>() -> T {
    let mut array = Array::<T>::new(2);
    array[0] = T::from_int(0);
    array[1] = T::from_int(1);
    array.to_vectorized(Comptime::new(UInt::new(2)))
}

#[cube]
pub fn array_of_one_to_vectorized_variable<T: Numeric>() -> T {
    let mut array = Array::<T>::new(1);
    array[0] = T::from_int(3);
    array.to_vectorized(Comptime::new(UInt::new(1)))
}

#[cube]
pub fn array_add_assign_simple(array: &mut Array<UInt>) {
    array[UInt::new(1)] += UInt::new(1);
}

#[cube]
pub fn array_add_assign_expr(array: &mut Array<UInt>) {
    array[UInt::new(1) + UInt::new(5)] += UInt::new(1);
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        ir::{Elem, Item, Variable},
    };

    type ElemType = F32;

    #[test]
    fn cube_support_array() {
        let mut context = CubeContext::root();

        array_read_write::__expand::<ElemType>(&mut context, 512);
        assert_eq!(
            format!("{:?}", context.into_scope().operations),
            inline_macro_ref_read_write()
        )
    }

    #[test]
    fn array_add_assign() {
        let mut context = CubeContext::root();
        let array = context.input(0, Item::new(Elem::UInt));

        array_add_assign_simple::__expand(&mut context, array.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_array_add_assign_simple()
        );
    }

    #[test]
    fn cube_array_to_vectorized() {
        let mut context = CubeContext::root();

        array_to_vectorized_variable::__expand::<ElemType>(&mut context);
        assert_eq!(
            format!("{:?}", context.into_scope().operations),
            inline_macro_ref_to_vectorized()
        );
    }

    #[test]
    fn cube_array_of_one_to_vectorized() {
        let mut context = CubeContext::root();

        array_of_one_to_vectorized_variable::__expand::<ElemType>(&mut context);
        assert_eq!(
            format!("{:?}", context.into_scope().operations),
            inline_macro_ref_one_to_vectorized()
        );
    }

    fn inline_macro_ref_read_write() -> String {
        let context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        let var = scope.create_local(item);
        let pos: Variable = 0u32.into();

        // Create
        let array = scope.create_local_array(item, 512);

        // Write
        cpa!(scope, array[pos] = 3.0_f32);

        // Read
        cpa!(scope, var = array[pos]);

        format!("{:?}", scope.operations)
    }

    #[test]
    fn array_add_assign_expr() {
        let mut context = CubeContext::root();
        let array = context.input(0, Item::new(Elem::UInt));

        array_add_assign_expr::__expand(&mut context, array.into());
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

    fn inline_macro_ref_to_vectorized() -> String {
        let context = CubeContext::root();
        let scalar_item = Item::new(ElemType::as_elem());
        let vectorized_item = Item::vectorized(ElemType::as_elem(), 2);

        let mut scope = context.into_scope();
        let pos0: Variable = 0u32.into();
        let pos1: Variable = 1u32.into();
        let array = scope.create_local_array(scalar_item, 2);
        cpa!(scope, array[pos0] = 0.0_f32);
        cpa!(scope, array[pos1] = 1.0_f32);

        let vectorized_var = scope.create_local(vectorized_item);
        let tmp = scope.create_local(scalar_item);
        cpa!(scope, tmp = array[pos0]);
        cpa!(scope, vectorized_var[pos0] = tmp);
        cpa!(scope, tmp = array[pos1]);
        cpa!(scope, vectorized_var[pos1] = tmp);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_one_to_vectorized() -> String {
        let context = CubeContext::root();
        let scalar_item = Item::new(ElemType::as_elem());
        let unvectorized_item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        let pos0: Variable = 0u32.into();
        let array = scope.create_local_array(scalar_item, 1);
        cpa!(scope, array[pos0] = 3.0_f32);

        let unvectorized_var = scope.create_local(unvectorized_item);
        let tmp = scope.create_local(scalar_item);
        cpa!(scope, tmp = array[pos0]);
        cpa!(scope, unvectorized_var = tmp);

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
