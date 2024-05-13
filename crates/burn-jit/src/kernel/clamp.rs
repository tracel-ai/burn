use super::unary;
use crate::{
    codegen::dialect::gpu::{ClampOperator, Operator, Scope, Variable},
    element::JitElement,
    tensor::JitTensor,
    unary, Runtime,
};

pub(crate) fn clamp<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    min_value: E,
    max_value: E,
) -> JitTensor<R, E, D> {
    unary!(
        operation: |scope: &mut Scope, elem, position: Variable| Operator::Clamp(ClampOperator {
            input: scope.read_array(0, elem, position),
            min_value: scope.read_scalar(0, elem),
            max_value: scope.read_scalar(1, elem),
            out: scope.create_local(elem),
        }),
        compiler: R::Compiler,
        scalar 2
    );

    unary::<Ops<R::Compiler, E>, OpsInplace<R::Compiler, E>, R, E, D>(
        input,
        Some(&[min_value, max_value]),
        true,
        Ops::new(),
        OpsInplace::new(),
    )
}
