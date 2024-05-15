use crate::{CubeContext, ExpandElement};
use burn_jit::gpu::{self, Elem, Variable};

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

pub(crate) fn cmp_expand<F>(
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

    let out_item = match lhs.item() {
        gpu::Item::Vec4(_) => gpu::Item::Vec4(Elem::Bool),
        gpu::Item::Vec3(_) => gpu::Item::Vec3(Elem::Bool),
        gpu::Item::Vec2(_) => gpu::Item::Vec2(Elem::Bool),
        gpu::Item::Scalar(_) => gpu::Item::Scalar(Elem::Bool),
    };
    let out = context.create_local(out_item);
    let out_var = *out;

    let op = func(gpu::BinaryOperator {
        lhs,
        rhs,
        out: out_var,
    });

    context.register(op);

    out
}

pub(crate) fn assign_op_expand<F>(
    context: &mut CubeContext,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(gpu::BinaryOperator) -> gpu::Operator,
{
    let lhs_var: Variable = *lhs;
    let rhs: Variable = *rhs;

    let op = func(gpu::BinaryOperator {
        lhs: lhs_var,
        rhs,
        out: lhs_var,
    });

    context.register(op);

    lhs
}
