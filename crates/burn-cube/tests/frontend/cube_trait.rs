use burn_cube::prelude::*;

#[cube]
trait FunctionGeneric {
    #[allow(unused)]
    fn test<C: Float>(lhs: C, rhs: C) -> C;
}

#[cube]
trait TraitGeneric<C: Float> {
    #[allow(unused)]
    fn test(lhs: C, rhs: C) -> C;
}

#[cube]
trait CombinedTraitFunctionGeneric<C: Float> {
    #[allow(unused)]
    fn test<O: Numeric>(lhs: C, rhs: C) -> O;
}

struct Test;

#[cube]
impl FunctionGeneric for Test {
    fn test<C: Float>(lhs: C, rhs: C) -> C {
        lhs + rhs
    }
}

#[cube]
impl<C: Float> TraitGeneric<C> for Test {
    fn test(lhs: C, rhs: C) -> C {
        lhs + rhs
    }
}

#[cube]
impl<C: Float> CombinedTraitFunctionGeneric<C> for Test {
    fn test<O: Numeric>(lhs: C, rhs: C) -> O {
        O::cast_from(lhs + rhs)
    }
}

#[cube]
fn simple<C: Float>(lhs: C, rhs: C) -> C {
    lhs + rhs
}

#[cube]
fn with_cast<C: Float, O: Numeric>(lhs: C, rhs: C) -> O {
    O::cast_from(lhs + rhs)
}

mod tests {
    use burn_cube::ir::{Item, Scope};

    use super::*;

    #[test]
    fn test_function_generic() {
        let mut context = CubeContext::root();
        let lhs = context.create_local(Item::new(F32::as_elem()));
        let rhs = context.create_local(Item::new(F32::as_elem()));

        <Test as FunctionGeneric>::test_expand::<F32>(&mut context, lhs, rhs);

        assert_eq!(simple_scope(), context.into_scope());
    }

    #[test]
    fn test_trait_generic() {
        let mut context = CubeContext::root();
        let lhs = context.create_local(Item::new(F32::as_elem()));
        let rhs = context.create_local(Item::new(F32::as_elem()));

        <Test as TraitGeneric<F32>>::test_expand(&mut context, lhs, rhs);

        assert_eq!(simple_scope(), context.into_scope());
    }

    #[test]
    fn test_combined_function_generic() {
        let mut context = CubeContext::root();
        let lhs = context.create_local(Item::new(F32::as_elem()));
        let rhs = context.create_local(Item::new(F32::as_elem()));

        <Test as CombinedTraitFunctionGeneric<F32>>::test_expand::<UInt>(&mut context, lhs, rhs);

        assert_eq!(with_cast_scope(), context.into_scope());
    }

    fn simple_scope() -> Scope {
        let mut context_ref = CubeContext::root();
        let lhs = context_ref.create_local(Item::new(F32::as_elem()));
        let rhs = context_ref.create_local(Item::new(F32::as_elem()));

        simple_expand::<F32>(&mut context_ref, lhs, rhs);
        context_ref.into_scope()
    }

    fn with_cast_scope() -> Scope {
        let mut context_ref = CubeContext::root();
        let lhs = context_ref.create_local(Item::new(F32::as_elem()));
        let rhs = context_ref.create_local(Item::new(F32::as_elem()));

        with_cast_expand::<F32, UInt>(&mut context_ref, lhs, rhs);
        context_ref.into_scope()
    }
}
