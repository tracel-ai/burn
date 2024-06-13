use burn_cube::prelude::*;

#[cube]
fn kernel<T: Numeric>(input: Tensor<T>) {
    let _shape = input.shape(1);
    let _stride = input.stride(1);
    let _length = input.len();
}

mod tests {
    use super::*;
    use burn_cube::{
        cpa,
        ir::{Item, Variable},
    };

    type ElemType = F32;

    #[test]
    fn cube_support_tensor_metadata() {
        let mut context = CubeContext::root();
        let input = context.input(0, Item::new(ElemType::as_elem()));

        kernel_expand::<ElemType>(&mut context, input);
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
