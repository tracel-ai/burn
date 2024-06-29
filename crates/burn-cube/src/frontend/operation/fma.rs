use crate::{
    ir::{FmaOperator, Operation, Operator},
    prelude::{CubeContext, CubePrimitive, ExpandElement},
    unexpanded,
};

/// Fused multiply-add `A*B+C`.
#[allow(unused_variables)]
pub fn fma<C: CubePrimitive>(a: C, b: C, c: C) -> C {
    unexpanded!()
}

/// Fused multiply-add and accumulate `C+=A*B`.
#[allow(unused_variables)]
pub fn fmaa<C: CubePrimitive>(a: C, b: C, c: &mut C) {
    unexpanded!()
}

/// Expand method of [fma].
#[allow(unused_variables)]
pub fn fma_expand<C: CubePrimitive>(
    context: &mut CubeContext,
    a: ExpandElement,
    b: ExpandElement,
    c: ExpandElement,
) -> ExpandElement {
    let output = context.create_local(a.item());

    let out = *output;
    let a = *a;
    let b = *b;
    let c = *c;

    context.register(Operation::Operator(Operator::Fma(FmaOperator {
        a,
        b,
        c,
        out,
    })));

    output
}

/// Expand method of [fma].
#[allow(unused_variables)]
pub fn fmaa_expand<C: CubePrimitive>(
    context: &mut CubeContext,
    a: ExpandElement,
    b: ExpandElement,
    c: ExpandElement,
) {
    let a = *a;
    let b = *b;
    let c = *c;

    context.register(Operation::Operator(Operator::Fma(FmaOperator {
        a,
        b,
        c,
        out: c,
    })));
}
