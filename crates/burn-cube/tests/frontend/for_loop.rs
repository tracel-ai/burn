use burn_cube::{
    cube,
    frontend::branch::range,
    frontend::{Array, Comptime, CubeContext, CubeElem, Float, UInt, F32},
};

type ElemType = F32;

#[cube]
pub fn for_loop<F: Float>(mut lhs: Array<F>, rhs: F, end: UInt, unroll: Comptime<bool>) {
    let tmp1 = rhs * rhs;
    let tmp2 = tmp1 + rhs;

    for i in range(0u32, end, unroll) {
        lhs[i] = tmp2 + lhs[i];
    }
}

mod tests {
    use burn_cube::{
        cpa,
        ir::{Item, Variable},
    };

    use super::*;

    #[test]
    fn test_for_loop_with_unroll() {
        let mut context = CubeContext::root();
        let unroll = true;

        let lhs = context.create_local(Item::new(ElemType::as_elem()));
        let rhs = context.create_local(Item::new(ElemType::as_elem()));
        let end = 4u32.into();

        for_loop_expand::<ElemType>(&mut context, lhs, rhs, end, unroll);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref(unroll));
    }

    #[test]
    fn test_for_loop_no_unroll() {
        let mut context = CubeContext::root();
        let unroll = false;

        let lhs = context.create_local(Item::new(ElemType::as_elem()));
        let rhs = context.create_local(Item::new(ElemType::as_elem()));
        let end = 4u32.into();

        for_loop_expand::<ElemType>(&mut context, lhs, rhs, end, unroll);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref(unroll));
    }

    fn inline_macro_ref(unroll: bool) -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());

        let lhs = context.create_local(item);
        let rhs = context.create_local(item);
        let lhs: Variable = lhs.into();
        let rhs: Variable = rhs.into();
        let end = 4u32;
        let mut scope = context.into_scope();

        // Kernel
        let tmp1 = scope.create_local(item);
        cpa!(scope, tmp1 = rhs * rhs);
        cpa!(scope, tmp1 = tmp1 + rhs);

        cpa!(
            &mut scope,
            range(0u32, end, unroll).for_each(|i, scope| {
                cpa!(scope, rhs = lhs[i]);
                cpa!(scope, rhs = tmp1 + rhs);
                cpa!(scope, lhs[i] = rhs);
            })
        );

        format!("{:?}", scope.operations)
    }
}
