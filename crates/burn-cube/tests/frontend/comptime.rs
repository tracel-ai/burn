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
pub fn comptime_else_then_if<T: Numeric>(lhs: T, cond1: Comptime<bool>, cond2: Comptime<bool>) {
    if Comptime::get(cond1) {
        let _ = lhs + T::from_int(4);
    } else {
        if Comptime::get(cond2) {
            let _ = lhs + T::from_int(5);
        } else {
            let _ = lhs - T::from_int(6);
        }
    }
}

#[cube]
pub fn comptime_elsif<T: Numeric>(lhs: T, cond1: Comptime<bool>, cond2: Comptime<bool>) {
    if Comptime::get(cond1) {
        let _ = lhs + T::from_int(4);
    } else if Comptime::get(cond2) {
        let _ = lhs + T::from_int(5);
    } else {
        let _ = lhs - T::from_int(6);
    }
}

#[cube]
pub fn comptime_elsif_with_runtime1<T: Numeric>(lhs: T, comptime_cond: Comptime<bool>) {
    let runtime_cond = lhs >= T::from_int(2);
    if Comptime::get(comptime_cond) {
        let _ = lhs + T::from_int(4);
    } else if runtime_cond {
        let _ = lhs + T::from_int(5);
    } else {
        let _ = lhs - T::from_int(6);
    }
}

#[cube]
pub fn comptime_elsif_with_runtime2<T: Numeric>(lhs: T, comptime_cond: Comptime<bool>) {
    let runtime_cond = lhs >= T::from_int(2);
    if runtime_cond {
        let _ = lhs + T::from_int(4);
    } else if Comptime::get(comptime_cond) {
        let _ = lhs + T::from_int(5);
    } else {
        let _ = lhs - T::from_int(6);
    }
}

#[cube]
pub fn comptime_if_expr<T: Numeric>(lhs: T, x: Comptime<UInt>, y: Comptime<UInt>) {
    let y2 = x + y;

    if x < y2 {
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
        frontend::{CubeContext, CubePrimitive, F32},
        ir::{Elem, Item, Variable},
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
    fn cube_comptime_if_numeric_test() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::new(ElemType::as_elem()));

        comptime_if_expr_expand::<ElemType>(&mut context, lhs, UInt::new(4), UInt::new(5));
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
    fn cube_comptime_elsif_test() {
        for cond1 in [false, true] {
            for cond2 in [false, true] {
                let mut context1 = CubeContext::root();
                let lhs = context1.create_local(Item::new(ElemType::as_elem()));
                comptime_else_then_if_expand::<ElemType>(&mut context1, lhs, cond1, cond2);
                let scope1 = context1.into_scope();

                let mut context2 = CubeContext::root();
                let lhs = context2.create_local(Item::new(ElemType::as_elem()));
                comptime_elsif_expand::<ElemType>(&mut context2, lhs, cond1, cond2);
                let scope2 = context2.into_scope();

                assert_eq!(
                    format!("{:?}", scope1.operations),
                    format!("{:?}", scope2.operations),
                );
            }
        }
    }

    #[test]
    fn cube_comptime_elsif_runtime1_test() {
        for cond in [false, true] {
            let mut context = CubeContext::root();
            let lhs = context.create_local(Item::new(ElemType::as_elem()));
            comptime_elsif_with_runtime1_expand::<ElemType>(&mut context, lhs, cond);
            let scope = context.into_scope();

            assert_eq!(
                format!("{:?}", scope.operations),
                inline_macro_ref_elsif_runtime1(cond)
            );
        }
    }

    #[test]
    fn cube_comptime_elsif_runtime2_test() {
        for cond in [false, true] {
            let mut context = CubeContext::root();
            let lhs = context.create_local(Item::new(ElemType::as_elem()));
            comptime_elsif_with_runtime2_expand::<ElemType>(&mut context, lhs, cond);
            let scope = context.into_scope();

            assert_eq!(
                format!("{:?}", scope.operations),
                inline_macro_ref_elsif_runtime2(cond)
            );
        }
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

    fn inline_macro_ref_elsif_runtime1(comptime_cond: bool) -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let runtime_cond = scope.create_local(Item::new(Elem::Bool));
        let y = scope.create_local(item);
        cpa!(scope, runtime_cond = x >= 2.0f32);

        if comptime_cond {
            cpa!(scope, y = x + 4.0f32);
        } else {
            cpa!(&mut scope, if(runtime_cond).then(|scope| {
                cpa!(scope, y = x + 5.0f32);
            }).else(|scope| {
                cpa!(scope, y = x - 6.0f32);
            }));
        };

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_elsif_runtime2(comptime_cond: bool) -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let runtime_cond = scope.create_local(Item::new(Elem::Bool));
        let y = scope.create_local(item);
        cpa!(scope, runtime_cond = x >= 2.0f32);

        cpa!(&mut scope, if(runtime_cond).then(|scope| {
            cpa!(scope, y = x + 4.0f32);
        }).else(|scope| {
            if comptime_cond {
                cpa!(scope, y = x + 5.0f32);
            } else {
                cpa!(scope, y = x - 6.0f32);
            }
        }));

        format!("{:?}", scope.operations)
    }
}
