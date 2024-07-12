use burn_cube::{
    cube,
    frontend::branch::range,
    frontend::{Array, Comptime, CubeContext, CubePrimitive, Float, UInt, F32},
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
    use burn_cube::{cpa, ir::Item};

    use super::*;

    #[test]
    fn test_for_loop_with_unroll() {
        let mut context = CubeContext::root();
        let unroll = true;

        let lhs = context.create_local_array(Item::new(ElemType::as_elem()), 4u32);
        let rhs = context.create_local(Item::new(ElemType::as_elem()));
        let end = 4u32.into();

        for_loop::__expand::<ElemType>(&mut context, lhs.into(), rhs, end, unroll);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref(unroll));
    }

    #[test]
    fn test_for_loop_no_unroll() {
        let mut context = CubeContext::root();
        let unroll = false;

        let lhs = context.create_local_array(Item::new(ElemType::as_elem()), 4u32);
        let rhs = context.create_local(Item::new(ElemType::as_elem()));
        let end = 4u32.into();

        for_loop::__expand::<ElemType>(&mut context, lhs.into(), rhs, end, unroll);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref(unroll));
    }

    fn inline_macro_ref(unroll: bool) -> String {
        let context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        let lhs = scope.create_local_array(item, 4u32);
        let rhs = scope.create_local(item);
        let end = 4u32;

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
