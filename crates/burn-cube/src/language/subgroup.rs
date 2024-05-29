use crate::{
    dialect::{Operation, Subgroup, SubgroupNoInput, UnaryOperator},
    unexpanded, CubeContext, CubeElem, ExpandElement,
};

/// Returns true if the cube unit has the lowest subgroup_unit_id among active unit in the subgroup
pub fn subgroup_elect() -> bool {
    unexpanded!()
}

pub fn subgroup_elect_expand<E: CubeElem>(context: &mut CubeContext) -> ExpandElement {
    let output = context.create_local(crate::dialect::Item::new(crate::dialect::Elem::Bool));

    let out = *output;

    context.register(Operation::Subgroup(Subgroup::SubgroupElect(
        SubgroupNoInput { out },
    )));

    output
}

pub fn subgroup_sum<E: CubeElem>(_elem: E) -> E {
    unexpanded!()
}

pub fn subgroup_sum_expand<E: CubeElem>(
    context: &mut CubeContext,
    elem: ExpandElement,
) -> ExpandElement {
    let output = context.create_local(elem.item());

    let out = *output;
    let input = *elem;

    context.register(Operation::Subgroup(Subgroup::SubgroupSum(UnaryOperator {
        input,
        out,
    })));

    output
}

pub fn subgroup_all<E: CubeElem>(_elem: E) -> E {
    unexpanded!()
}

pub fn subgroup_all_expand<E: CubeElem>(
    context: &mut CubeContext,
    elem: ExpandElement,
) -> ExpandElement {
    let output = context.create_local(elem.item());

    let out = *output;
    let input = *elem;

    context.register(Operation::Subgroup(Subgroup::SubgroupAll(UnaryOperator {
        input,
        out,
    })));

    output
}
