use crate::frontend::{CubeContext, ExpandElement};
use crate::ir::{BinaryOperator, Elem, Item, Operator, UnaryOperator, Variable, Vectorization};

pub(crate) fn binary_expand<F>(
    context: &mut CubeContext,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Operator,
{
    let lhs_var: Variable = *lhs;
    let rhs_var: Variable = *rhs;

    let item_lhs = lhs.item();
    let item_rhs = rhs.item();

    let vectorization = check_vectorization(item_lhs.vectorization, item_rhs.vectorization);
    let item = Item::vectorized(item_lhs.elem, vectorization);

    // We can only reuse rhs.
    let out = if lhs.can_mut() && item_lhs == item {
        lhs
    } else if rhs.can_mut() && item_rhs == item {
        rhs
    } else {
        context.create_local(item)
    };

    let out_var = *out;

    let op = func(BinaryOperator {
        lhs: lhs_var,
        rhs: rhs_var,
        out: out_var,
    });

    context.register(op);

    out
}

pub(crate) fn binary_expand_no_vec<F>(
    context: &mut CubeContext,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Operator,
{
    let lhs_var: Variable = *lhs;
    let rhs_var: Variable = *rhs;

    let item_lhs = lhs.item();
    let item_rhs = rhs.item();

    let item = Item::new(item_lhs.elem);

    // We can only reuse rhs.
    let out = if lhs.can_mut() && item_lhs == item {
        lhs
    } else if rhs.can_mut() && item_rhs == item {
        rhs
    } else {
        context.create_local(item)
    };

    let out_var = *out;

    let op = func(BinaryOperator {
        lhs: lhs_var,
        rhs: rhs_var,
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
    F: Fn(BinaryOperator) -> Operator,
{
    let lhs: Variable = *lhs;
    let rhs: Variable = *rhs;
    let item = lhs.item();

    check_vectorization(item.vectorization, rhs.item().vectorization);

    let out_item = Item {
        elem: Elem::Bool,
        vectorization: item.vectorization,
    };

    let out = context.create_local(out_item);
    let out_var = *out;

    let op = func(BinaryOperator {
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
    F: Fn(BinaryOperator) -> Operator,
{
    let lhs_var: Variable = *lhs;
    let rhs: Variable = *rhs;

    check_vectorization(lhs_var.item().vectorization, rhs.item().vectorization);

    let op = func(BinaryOperator {
        lhs: lhs_var,
        rhs,
        out: lhs_var,
    });

    context.register(op);

    lhs
}

pub fn unary_expand<F>(context: &mut CubeContext, input: ExpandElement, func: F) -> ExpandElement
where
    F: Fn(UnaryOperator) -> Operator,
{
    let input_var: Variable = *input;

    let item = input.item();

    let out = if input.can_mut() {
        input
    } else {
        context.create_local(item)
    };

    let out_var = *out;

    let op = func(UnaryOperator {
        input: input_var,
        out: out_var,
    });

    context.register(op);

    out
}

pub fn init_expand<F>(context: &mut CubeContext, input: ExpandElement, func: F) -> ExpandElement
where
    F: Fn(UnaryOperator) -> Operator,
{
    if input.can_mut() {
        return input;
    }

    let input_var: Variable = *input;
    let item = input.item();

    let out = context.create_local(item);
    let out_var = *out;

    let op = func(UnaryOperator {
        input: input_var,
        out: out_var,
    });

    context.register(op);

    out
}

fn check_vectorization(lhs: Vectorization, rhs: Vectorization) -> Vectorization {
    let output = u8::max(lhs, rhs);

    if lhs == 1 || rhs == 1 {
        return output;
    }

    assert!(
        lhs == rhs,
        "Tried to perform binary operation on different vectorization schemes."
    );

    output
}

pub fn array_assign_binary_op_expand<
    Array: Into<ExpandElement>,
    Index: Into<ExpandElement>,
    Value: Into<ExpandElement>,
    F: Fn(BinaryOperator) -> Operator,
>(
    context: &mut CubeContext,
    array: Array,
    index: Index,
    value: Value,
    func: F,
) {
    let array: ExpandElement = array.into();
    let index: ExpandElement = index.into();
    let value: ExpandElement = value.into();

    let tmp = context.create_local(array.item());

    let read = Operator::Index(BinaryOperator {
        lhs: *array,
        rhs: *index,
        out: *tmp,
    });
    let calculate = func(BinaryOperator {
        lhs: *tmp,
        rhs: *value,
        out: *tmp,
    });

    let write = Operator::IndexAssign(BinaryOperator {
        lhs: *index,
        rhs: *tmp,
        out: *array,
    });

    context.register(read);
    context.register(calculate);
    context.register(write);
}
