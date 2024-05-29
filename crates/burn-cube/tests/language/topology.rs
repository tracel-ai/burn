use burn_cube::{cube, Numeric, Tensor, UInt, ABSOLUTE_INDEX};

#[cube]
fn topology_kernel<T: Numeric>(input: Tensor<T>) {
    let id = ABSOLUTE_INDEX + UInt::new(4u32);
    let _ = input[id];
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        dialect::{Item, Variable},
        CubeContext, CubeElem, UInt, F32,
    };

    type ElemType = F32;

    #[test]
    fn cube_support_topology() {
        let mut context = CubeContext::root();
        let input = context.input(0, Item::new(ElemType::as_elem()));

        topology_kernel_expand::<ElemType>(&mut context, input);
        assert_eq!(
            format!("{:?}", context.into_scope().operations),
            inline_macro_ref()
        );
    }

    fn inline_macro_ref() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());
        let input = context.input(0, item);

        let mut scope = context.into_scope();
        let input: Variable = input.into();
        let x = scope.create_local(Item::new(UInt::as_elem()));
        let y = scope.create_local(Item::new(UInt::as_elem()));
        let z = scope.create_local(Item::new(UInt::as_elem()));

        cpa!(&mut scope, x = shape(input, 1u32));
        cpa!(&mut scope, y = stride(input, 1u32));
        cpa!(&mut scope, z = len(input));

        format!("{:?}", scope.operations)
    }
}
