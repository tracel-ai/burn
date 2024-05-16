use burn_cube::{cube, CubeContext, Numeric, PrimitiveVariable, UInt, I64};
use burn_jit::gpu::{Elem, Item};

#[cube]
pub fn caller_no_arg(x: UInt) {
    let _ = x + callee_no_arg();
}

#[cube]
pub fn callee_no_arg() -> UInt {
    UInt::lit(8)
}

#[cube]
pub fn no_call_no_arg(x: UInt) {
    let _ = x + UInt::lit(8);
}

#[cube]
pub fn caller_with_arg(x: UInt) {
    let _ = x + callee_with_arg(x);
}

#[cube]
pub fn callee_with_arg(x: UInt) -> UInt {
    x * UInt::lit(8)
}

#[cube]
pub fn no_call_with_arg(x: UInt) {
    let _ = x + x * UInt::lit(8);
}

#[cube]
pub fn caller_with_generics<T: Numeric>(x: T) {
    let _ = x + callee_with_generics::<T>(x);
}

#[cube]
pub fn callee_with_generics<T: Numeric>(x: T) -> T {
    x * T::lit(8)
}

#[cube]
pub fn no_call_with_generics<T: Numeric>(x: T) {
    let _ = x + x * T::lit(8);
}

#[test]
fn cube_call_equivalent_to_no_call_no_arg_test() {
    let mut caller_context = CubeContext::root();
    let x = caller_context.create_local(Item::Scalar(Elem::UInt));
    caller_no_arg_expand(&mut caller_context, x);
    let caller_scope = caller_context.into_scope();

    let mut no_call_context = CubeContext::root();
    let x = no_call_context.create_local(Item::Scalar(Elem::UInt));
    no_call_no_arg_expand(&mut no_call_context, x);
    let no_call_scope = no_call_context.into_scope();

    assert_eq!(
        format!("{:?}", caller_scope.operations),
        format!("{:?}", no_call_scope.operations)
    );
}

#[test]
fn cube_call_equivalent_to_no_call_with_arg_test() {
    let mut caller_context = CubeContext::root();

    let x = caller_context.create_local(Item::Scalar(Elem::UInt));
    caller_with_arg_expand(&mut caller_context, x);
    let caller_scope = caller_context.into_scope();

    let mut no_call_context = CubeContext::root();
    let x = no_call_context.create_local(Item::Scalar(Elem::UInt));
    no_call_with_arg_expand(&mut no_call_context, x);
    let no_call_scope = no_call_context.into_scope();

    assert_eq!(
        format!("{:?}", caller_scope.operations),
        format!("{:?}", no_call_scope.operations)
    );
}

#[test]
fn cube_call_equivalent_to_no_call_with_generics_test() {
    let mut caller_context = CubeContext::root();
    type ElemType = I64;
    let x = caller_context.create_local(Item::Scalar(ElemType::into_elem()));
    caller_with_generics_expand::<ElemType>(&mut caller_context, x);
    let caller_scope = caller_context.into_scope();

    let mut no_call_context = CubeContext::root();
    let x = no_call_context.create_local(Item::Scalar(ElemType::into_elem()));
    no_call_with_generics_expand::<ElemType>(&mut no_call_context, x);
    let no_call_scope = no_call_context.into_scope();

    assert_eq!(
        format!("{:?}", caller_scope.operations),
        format!("{:?}", no_call_scope.operations)
    );
}
