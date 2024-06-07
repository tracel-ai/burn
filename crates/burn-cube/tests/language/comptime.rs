use burn_cube::prelude::*;

#[derive(Clone)]
pub struct State {
    cond: bool,
    bound: u32,
}

impl Init for State {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[cube]
pub fn comptime_if_else<T: Numeric>(lhs: T, cond: Comptime<bool>) {
    if Comptime::get(cond) {
        let _ = lhs + T::from_int(4);
    } else {
        let _ = lhs - T::from_int(5);
    }
}

#[cube]
pub fn comptime_with_map_bool<T: Numeric>(state: Comptime<State>) -> T {
    let cond = Comptime::map(state, |s: State| s.cond);

    let mut x = T::from_int(3);
    if Comptime::get(cond) {
        x += T::from_int(4);
    } else {
        x -= T::from_int(4);
    }
    x
}

#[cube]
pub fn comptime_with_map_uint<T: Numeric>(state: Comptime<State>) -> T {
    let bound = Comptime::map(state, |s: State| s.bound);

    let mut x = T::from_int(3);
    for _ in range(0u32, Comptime::get(bound), Comptime::new(true)) {
        x += T::from_int(4);
    }

    x
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        frontend::{CubeContext, CubeElem, F32},
        ir::{Item, Variable},
    };

    type ElemType = F32;

    #[test]
    fn cube_comptime_if_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(ElemType::as_elem()));

        comptime_if_else_expand::<ElemType>(&mut context, lhs, true);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_comptime(true)
        );
    }

    #[test]
    fn cube_comptime_else_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(ElemType::as_elem()));

        comptime_if_else_expand::<ElemType>(&mut context, lhs, false);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_comptime(false)
        );
    }

    #[test]
    fn cube_comptime_map_bool_test() {
        let mut context1 = CubeContext::root();
        let mut context2 = CubeContext::root();

        let comptime_state_true = State {
            cond: true,
            bound: 4,
        };
        let comptime_state_false = State {
            cond: false,
            bound: 4,
        };

        comptime_with_map_bool_expand::<ElemType>(&mut context1, comptime_state_true);
        comptime_with_map_bool_expand::<ElemType>(&mut context2, comptime_state_false);

        let scope1 = context1.into_scope();
        let scope2 = context2.into_scope();

        assert_ne!(
            format!("{:?}", scope1.operations),
            format!("{:?}", scope2.operations)
        );
    }

    #[test]
    fn cube_comptime_map_uint_test() {
        let mut context = CubeContext::root();

        let comptime_state = State {
            cond: true,
            bound: 4,
        };

        comptime_with_map_uint_expand::<ElemType>(&mut context, comptime_state);

        let scope = context.into_scope();

        assert!(!format!("{:?}", scope.operations).contains("RangeLoop"));
    }

    fn inline_macro_ref_comptime(cond: bool) -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let y = scope.create_local(item);

        if cond {
            cpa!(scope, y = x + 4.0f32);
        } else {
            cpa!(scope, y = x - 5.0f32);
        };

        format!("{:?}", scope.operations)
    }
}
