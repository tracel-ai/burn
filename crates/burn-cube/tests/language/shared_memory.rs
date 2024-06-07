use burn_cube::prelude::*;

// #[cube(debug)]
// fn shared_memory_read_write<T: Numeric>(sm_size: Comptime<u32>) {
//     let mut shared = SharedMemory::<T>::new(sm_size);
//     shared[0] = T::from_int(3);
//     let _ = shared[0];
// }

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn shared_memory_read_write<T: Numeric>(sm_size: Comptime<u32>) {
    let mut shared = SharedMemory::<T>::new(sm_size);
    shared[0] = T::from_int(3);
    let _ = shared[0];
}
#[allow(unused_mut)]
#[allow(clippy::too_many_arguments)]
#[doc = r" Expanded Cube function"]
pub fn shared_memory_read_write_expand<T: Numeric>(
    context: &mut burn_cube::frontend::CubeContext,
    sm_size: <Comptime<u32> as burn_cube::frontend::CubeType>::ExpandType,
) {
    let mut shared = {
        let _inner = {
            let _var_0 = sm_size;
            SharedMemory::<T>::new_expand(context, _var_0)
        };
        burn_cube::frontend::Init::init(_inner, context)
    };
    {
        let _array = shared.clone();
        let _index = 0;
        let _value = {
            let _var_0 = 3;
            T::from_int_expand(context, _var_0)
        };
        burn_cube::frontend::index_assign::expand(context, _array, _index, _value)
    };
    let _ = {
        let _inner = {
            let _array = shared;
            let _index = 0;
            burn_cube::frontend::index::expand(context, _array, _index)
        };
        burn_cube::frontend::Init::init(_inner, context)
    };
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        ir::{Item, Variable},
    };

    type ElemType = F32;

    #[test]
    fn cube_support_shared_memory() {
        let mut context = CubeContext::root();

        shared_memory_read_write_expand::<ElemType>(&mut context, 512);
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
        let shared = scope.create_shared(item, 512);

        // Write
        cpa!(scope, shared[pos] = 3.0_f32);

        // Read
        cpa!(scope, var = shared[pos]);

        format!("{:?}", scope.operations)
    }
}
