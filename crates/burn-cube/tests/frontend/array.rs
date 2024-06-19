use burn_cube::prelude::*;

#[cube]
fn array_read_write<T: Numeric>(array_size: Comptime<u32>) {
    let mut array = Array::<T>::new(array_size);
    array[0] = T::from_int(3);
    let _ = array[0];
}

#[cube]
fn array_to_vectorized_variable<T: Numeric>() -> T {
    let mut array = Array::<T>::new(2);
    array[0] = T::from_int(0);
    array[1] = T::from_int(1);
    array.to_vectorized(Comptime::new(UInt::new(2)))
}

#[cube]
fn array_of_one_to_vectorized_variable<T: Numeric>() -> T {
    let mut array = Array::<T>::new(1);
    array[0] = T::from_int(3);
    array.to_vectorized(Comptime::new(UInt::new(1)))
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        ir::{Item, Variable},
    };

    type ElemType = F32;

    #[test]
    fn cube_support_array() {
        let mut context = CubeContext::root();

        array_read_write_expand::<ElemType>(&mut context, 512);
        assert_eq!(
            format!("{:?}", context.into_scope().operations),
            inline_macro_ref_read_write()
        );
    }

    #[test]
    fn cube_array_to_vectorized() {
        let mut context = CubeContext::root();

        array_to_vectorized_variable_expand::<ElemType>(&mut context);
        assert_eq!(
            format!("{:?}", context.into_scope().operations),
            inline_macro_ref_to_vectorized()
        );
    }

    #[test]
    fn cube_array_of_one_to_vectorized() {
        let mut context = CubeContext::root();

        array_of_one_to_vectorized_variable_expand::<ElemType>(&mut context);
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
}
