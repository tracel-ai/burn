use crate::{CubeContext, ExpandElement};
use burn_jit::gpu::{self, Variable};

pub(crate) fn binary_expand<F>(
    context: &mut CubeContext,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(gpu::BinaryOperator) -> gpu::Operator,
{
    let lhs: Variable = *lhs;
    let rhs: Variable = *rhs;

    let item = lhs.item();
    let out = context.create_local(item);
    let out_var = *out;

    let op = func(gpu::BinaryOperator {
        lhs,
        rhs,
        out: out_var,
    });

    context.register(op);

    out
}
