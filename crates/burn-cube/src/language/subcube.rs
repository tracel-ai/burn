use crate::{
    dialect::{Operation, Subcube, SubgroupNoInput, UnaryOperator},
    unexpanded, CubeContext, CubeElem, ExpandElement,
};

/// Returns true if the cube unit has the lowest subcube_unit_id among active unit in the subcube
pub fn subcube_elect() -> bool {
    unexpanded!()
}

pub fn subcube_elect_expand<E: CubeElem>(context: &mut CubeContext) -> ExpandElement {
    let output = context.create_local(crate::dialect::Item::new(crate::dialect::Elem::Bool));

    let out = *output;

    context.register(Operation::Subcube(Subcube::SubcubeElect(SubgroupNoInput {
        out,
    })));

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

    context.register(Operation::Subcube(Subcube::SubcubeSum(UnaryOperator {
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

    context.register(Operation::Subcube(Subcube::SubcubeAll(UnaryOperator {
        input,
        out,
    })));

    output
}
