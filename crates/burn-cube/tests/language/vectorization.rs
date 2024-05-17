use burn_cube::{cube, Numeric};

#[cube]
pub fn vectorization<T: Numeric>(lhs: T) {
    let _ = lhs + T::from_vec(&[4, 5]);
}

mod tests {

    use burn_cube::{dialect::Item, CubeContext, PrimitiveVariable, F32};

    use crate::language::vectorization::vectorization_expand;

    type ElemType = F32;

    #[test]
    fn cube_vectorization_with_same_scheme_does_not_fail() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::vectorized(ElemType::into_elem(), 2));

        vectorization_expand::<ElemType>(&mut context, lhs);
    }

    #[test]
    #[should_panic]
    fn cube_vectorization_with_different_scheme_fails() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::vectorized(ElemType::into_elem(), 4));

        vectorization_expand::<ElemType>(&mut context, lhs);
    }
}
