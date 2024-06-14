use burn_cube::prelude::*;

#[cube]
fn array_read_write<T: Numeric>(array_size: Comptime<u32>) {
    let mut array = Array::<T>::new(array_size);
    array[0] = T::from_int(3);
    let _ = array[0];
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
            inline_macro_ref()
        );
    }

    fn inline_macro_ref() -> String {
        let context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        let var = scope.create_local(item);
        let pos: Variable = 0u32.into();

        // Create
        let shared = scope.create_local_array(item, 512);

        // Write
        cpa!(scope, shared[pos] = 3.0_f32);

        // Read
        cpa!(scope, var = shared[pos]);

        format!("{:?}", scope.operations)
    }
}
