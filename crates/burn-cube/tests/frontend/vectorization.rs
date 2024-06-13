use burn_cube::prelude::*;

#[cube]
pub fn vectorization_binary<T: Numeric>(lhs: T) {
    let _ = lhs + T::from_vec(&[4, 5]);
}

#[cube]
pub fn vectorization_cmp<T: Numeric>(rhs: T) {
    let _ = T::from_vec(&[4, 5]) > rhs;
}

mod tests {
    use super::*;
    use burn_cube::ir::Item;

    type ElemType = F32;

    #[test]
    fn cube_vectorization_binary_op_with_same_scheme_does_not_fail() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::vectorized(ElemType::as_elem(), 2));

        vectorization_binary_expand::<ElemType>(&mut context, lhs);
    }

    #[test]
    #[should_panic]
    fn cube_vectorization_binary_op_with_different_scheme_fails() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::vectorized(ElemType::as_elem(), 4));

        vectorization_binary_expand::<ElemType>(&mut context, lhs);
    }

    #[test]
    fn cube_vectorization_cmp_op_with_same_scheme_does_not_fail() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::vectorized(ElemType::as_elem(), 2));

        vectorization_cmp_expand::<ElemType>(&mut context, lhs);
    }

    #[test]
    #[should_panic]
    fn cube_vectorization_cmp_op_with_different_scheme_fails() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::vectorized(ElemType::as_elem(), 4));

        vectorization_cmp_expand::<ElemType>(&mut context, lhs);
    }

    #[test]
    fn cube_vectorization_can_be_broadcasted() {
        let mut context = CubeContext::root();

        let lhs = context.create_local(Item::vectorized(ElemType::as_elem(), 1));

        vectorization_cmp_expand::<ElemType>(&mut context, lhs);
    }
}
