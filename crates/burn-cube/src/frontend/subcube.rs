use super::{CubeContext, CubeElem, ExpandElement};
use crate::{
    ir::{Elem, InitOperator, Item, Operation, Subcube, UnaryOperator},
    unexpanded,
};

/// Returns true if the cube unit has the lowest subcube_unit_id among active unit in the subcube
pub fn subcube_elect() -> bool {
    unexpanded!()
}

pub fn subcube_elect_expand<E: CubeElem>(context: &mut CubeContext) -> ExpandElement {
    let output = context.create_local(Item::new(Elem::Bool));

    let out = *output;

    context.register(Operation::Subcube(Subcube::Elect(InitOperator { out })));

    output
}

pub fn subcube_sum<E: CubeElem>(_elem: E) -> E {
    unexpanded!()
}

pub fn subcube_sum_expand<E: CubeElem>(
    context: &mut CubeContext,
    elem: ExpandElement,
) -> ExpandElement {
    let output = context.create_local(elem.item());

    let out = *output;
    let input = *elem;

    context.register(Operation::Subcube(Subcube::Sum(UnaryOperator {
        input,
        out,
    })));

    output
}

pub fn subcube_prod<E: CubeElem>(_elem: E) -> E {
    unexpanded!()
}

pub fn subcube_prod_expand<E: CubeElem>(
    context: &mut CubeContext,
    elem: ExpandElement,
) -> ExpandElement {
    let output = context.create_local(elem.item());

    let out = *output;
    let input = *elem;

    context.register(Operation::Subcube(Subcube::Prod(UnaryOperator {
        input,
        out,
    })));

    output
}

pub fn subcube_max<E: CubeElem>(_elem: E) -> E {
    unexpanded!()
}

pub fn subcube_max_expand<E: CubeElem>(
    context: &mut CubeContext,
    elem: ExpandElement,
) -> ExpandElement {
    let output = context.create_local(elem.item());

    let out = *output;
    let input = *elem;

    context.register(Operation::Subcube(Subcube::Max(UnaryOperator {
        input,
        out,
    })));

    output
}

pub fn subcube_min<E: CubeElem>(_elem: E) -> E {
    unexpanded!()
}

pub fn subcube_min_expand<E: CubeElem>(
    context: &mut CubeContext,
    elem: ExpandElement,
) -> ExpandElement {
    let output = context.create_local(elem.item());

    let out = *output;
    let input = *elem;

    context.register(Operation::Subcube(Subcube::Min(UnaryOperator {
        input,
        out,
    })));

    output
}

pub fn subcube_all<E: CubeElem>(_elem: E) -> E {
    unexpanded!()
}

pub fn subcube_all_expand<E: CubeElem>(
    context: &mut CubeContext,
    elem: ExpandElement,
) -> ExpandElement {
    let output = context.create_local(elem.item());

    let out = *output;
    let input = *elem;

    context.register(Operation::Subcube(Subcube::All(UnaryOperator {
        input,
        out,
    })));

    output
}
