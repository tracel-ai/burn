use burn_cube::prelude::*;

#[derive(CubeType)]
struct State<T: Numeric> {
    first: T,
    second: T,
}

#[cube]
fn state_receiver_with_reuse<T: Numeric>(state: State<T>) -> T {
    let x = state.first + state.second;
    state.second + x + state.first
}

#[cube]
fn attribute_modifier_reuse_field<T: Numeric>(mut state: State<T>) -> T {
    state.first = T::from_int(4);
    state.first
}

#[cube]
fn attribute_modifier_reuse_struct<T: Numeric>(mut state: State<T>) -> State<T> {
    state.first = T::from_int(4);
    state
}

#[cube]
fn creator<T: Numeric>(x: T, second: T) -> State<T> {
    let mut state = State::<T> { first: x, second };
    state.second = state.first;

    state
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        ir::{Item, Variable},
    };

    type ElemType = F32;

    #[test]
    fn cube_new_struct_test() {
        let mut context = CubeContext::root();

        let x = context.create_local(Item::new(ElemType::as_elem()));
        let y = context.create_local(Item::new(ElemType::as_elem()));

        creator_expand::<ElemType>(&mut context, x, y);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            creator_inline_macro_ref()
        );
    }

    #[test]
    fn cube_struct_as_arg_test() {
        let mut context = CubeContext::root();

        let x = context.create_local(Item::new(ElemType::as_elem()));
        let y = context.create_local(Item::new(ElemType::as_elem()));

        let expanded_state = StateExpand {
            first: x,
            second: y,
        };
        state_receiver_with_reuse_expand::<ElemType>(&mut context, expanded_state);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            receive_state_with_reuse_inline_macro_ref()
        );
    }

    #[test]
    fn cube_struct_assign_to_field_test() {
        let mut context = CubeContext::root();

        let x = context.create_local(Item::new(ElemType::as_elem()));
        let y = context.create_local(Item::new(ElemType::as_elem()));

        let expanded_state = StateExpand {
            first: x,
            second: y,
        };
        attribute_modifier_reuse_field_expand::<ElemType>(&mut context, expanded_state);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            field_modifier_inline_macro_ref()
        );
    }

    #[test]
    fn cube_struct_assign_to_field_reuse_struct_test() {
        let mut context = CubeContext::root();

        let x = context.create_local(Item::new(ElemType::as_elem()));
        let y = context.create_local(Item::new(ElemType::as_elem()));

        let expanded_state = StateExpand {
            first: x,
            second: y,
        };
        attribute_modifier_reuse_struct_expand::<ElemType>(&mut context, expanded_state);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            field_modifier_inline_macro_ref()
        );
    }

    fn creator_inline_macro_ref() -> String {
        let context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        let x = scope.create_local(item);
        let y = scope.create_local(item);
        cpa!(scope, y = x);

        format!("{:?}", scope.operations)
    }

    fn field_modifier_inline_macro_ref() -> String {
        let context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        scope.create_with_value(4, item);

        format!("{:?}", scope.operations)
    }

    fn receive_state_with_reuse_inline_macro_ref() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local(item);
        let y = context.create_local(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let y: Variable = y.into();
        let z = scope.create_local(item);

        cpa!(scope, z = x + y);
        cpa!(scope, z = y + z);
        cpa!(scope, z = z + x);

        format!("{:?}", scope.operations)
    }
}
