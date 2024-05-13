use burn_cube::{cube, CubeContext, CubeType, Float, Numeric, PrimitiveVariable, F32};
use burn_jit::{
    gpu,
    gpu::{Item, Variable},
};

type ElemType = F32;

/// Traits used in Cube kernels must expose an _expand variant
/// for all their methods. However, one does not need to provide its
/// implementation, see examples below.
trait Strategy<T: Numeric> {
    fn operation(input_1: T, input_2: T) -> T;
    fn operation_expand(
        context: &mut CubeContext,
        input_1: <T as CubeType>::ExpandType,
        input_2: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType;
}

struct AddStrategy;

#[cube]
/// The actual implementation of AddStrategy's operation
/// Automatically generated an _expand variant
fn add_strategy_operation<T: Numeric>(input_1: T, input_2: T) -> T {
    input_1 + input_2
}

impl<T: Numeric> Strategy<T> for AddStrategy {
    /// Here we link the trait's method to the cube function
    fn operation(input_1: T, input_2: T) -> T {
        add_strategy_operation(input_1, input_2)
    }

    /// Here we link the trait's expanded method to the cube expanded function
    fn operation_expand(
        context: &mut CubeContext,
        input_1: <T as CubeType>::ExpandType,
        input_2: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        add_strategy_operation_expand::<T>(context, input_1, input_2)
    }
}

struct SubStrategy;

#[cube]
fn sub_strategy_operation<T: Numeric>(input_1: T, input_2: T) -> T {
    input_1 - input_2
}

impl<T: Numeric> Strategy<T> for SubStrategy {
    fn operation(input_1: T, input_2: T) -> T {
        sub_strategy_operation(input_1, input_2)
    }

    fn operation_expand(
        context: &mut CubeContext,
        input_1: <T as CubeType>::ExpandType,
        input_2: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        sub_strategy_operation_expand::<T>(context, input_1, input_2)
    }
}

#[cube]
fn with_strategy_trait<S: Strategy<T>, T: Numeric>(x: T, y: T) -> T {
    S::operation(x, y)
}

#[cube]
fn two_strategy_traits<S1: Strategy<F>, S2: Strategy<F>, F: Float>(x: F, y: F) -> F {
    let z = S1::operation(x, y);
    S2::operation(z, y)
}

trait MethodTypedStrategy {
    fn operation<T: Numeric>(input_1: T, input_2: T) -> T;
    fn operation_expand<T: Numeric>(
        _context: &mut CubeContext,
        input_1: <T as CubeType>::ExpandType,
        input_2: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType;
}

impl MethodTypedStrategy for AddStrategy {
    fn operation<T: Numeric>(input_1: T, input_2: T) -> T {
        add_strategy_operation(input_1, input_2)
    }

    fn operation_expand<T: Numeric>(
        context: &mut CubeContext,
        input_1: <T as CubeType>::ExpandType,
        input_2: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        add_strategy_operation_expand::<T>(context, input_1, input_2)
    }
}

#[cube]
fn with_trait_generic_method<S: MethodTypedStrategy, T: Numeric>(x: T, y: T) -> T {
    S::operation::<T>(x, y)
}

#[test]
fn cube_strategy_trait_add_test() {
    let mut context = CubeContext::root();

    let x = context.create_local(Item::Scalar(ElemType::into_elem()));
    let y = context.create_local(Item::Scalar(ElemType::into_elem()));

    with_strategy_trait_expand::<AddStrategy, ElemType>(&mut context, x, y);
    let scope = context.into_scope();

    assert_eq!(
        format!("{:?}", scope.operations),
        inline_macro_ref_one(true)
    );
}

#[test]
fn cube_strategy_trait_sub_test() {
    let mut context = CubeContext::root();

    let x = context.create_local(Item::Scalar(ElemType::into_elem()));
    let y = context.create_local(Item::Scalar(ElemType::into_elem()));

    with_strategy_trait_expand::<SubStrategy, ElemType>(&mut context, x, y);
    let scope = context.into_scope();

    assert_eq!(
        format!("{:?}", scope.operations),
        inline_macro_ref_one(false)
    );
}

#[test]
fn cube_two_strategy_traits_test() {
    let mut context = CubeContext::root();

    let x = context.create_local(Item::Scalar(ElemType::into_elem()));
    let y = context.create_local(Item::Scalar(ElemType::into_elem()));

    two_strategy_traits_expand::<SubStrategy, AddStrategy, ElemType>(&mut context, x, y);
    let scope = context.into_scope();

    assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_two());
}

#[test]
fn cube_trait_generic_method_test() {
    let mut context = CubeContext::root();

    let x = context.create_local(Item::Scalar(ElemType::into_elem()));
    let y = context.create_local(Item::Scalar(ElemType::into_elem()));

    with_trait_generic_method_expand::<AddStrategy, ElemType>(&mut context, x, y);
    let scope = context.into_scope();

    assert_eq!(
        format!("{:?}", scope.operations),
        inline_macro_ref_one(true)
    );
}

fn inline_macro_ref_one(is_add_strategy: bool) -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(ElemType::into_elem());
    let x = context.create_local(item);
    let y = context.create_local(item);

    let mut scope = context.into_scope();
    let x: Variable = x.into();
    let y: Variable = y.into();
    let tmp = scope.create_local(item);

    match is_add_strategy {
        true => gpu!(scope, tmp = x + y),
        false => gpu!(scope, tmp = x - y),
    }

    format!("{:?}", scope.operations)
}

fn inline_macro_ref_two() -> String {
    let mut context = CubeContext::root();
    let item = Item::Scalar(ElemType::into_elem());
    let x = context.create_local(item);
    let y = context.create_local(item);

    let mut scope = context.into_scope();
    let x: Variable = x.into();
    let y: Variable = y.into();
    let tmp = scope.create_local(item);

    gpu!(scope, tmp = x - y);
    gpu!(scope, x = tmp + y);

    format!("{:?}", scope.operations)
}
