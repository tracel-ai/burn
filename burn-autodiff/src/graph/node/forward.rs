use std::sync::Arc;

use super::ForwardNodeState;
use crate::graph::ops::ForwardRecordedOpsBoxed;

use burn_common::id::IdGenerator;

#[derive(Debug)]
pub struct ForwardNode<Out> {
    pub id: String,
    pub order: usize,
    pub state: ForwardNodeState<Out>,
    pub ops: ForwardRecordedOpsBoxed<Out>,
}
pub type ForwardNodeRef<Out> = Arc<ForwardNode<Out>>;

impl<Out> ForwardNode<Out> {
    pub fn from_root(state: ForwardNodeState<Out>, ops: ForwardRecordedOpsBoxed<Out>) -> Self {
        let order = 0;
        Self::new(order, state, ops)
    }

    pub fn from_unary<T>(
        node: &ForwardNode<T>,
        state: ForwardNodeState<Out>,
        ops: ForwardRecordedOpsBoxed<Out>,
    ) -> Self {
        let order = node.order + 1;
        Self::new(order, state, ops)
    }

    pub fn from_binary<Lhs, Rhs>(
        lhs: &ForwardNode<Lhs>,
        rhs: &ForwardNode<Rhs>,
        state: ForwardNodeState<Out>,
        ops: ForwardRecordedOpsBoxed<Out>,
    ) -> Self {
        let order = usize::max(lhs.order, rhs.order) + 1;
        Self::new(order, state, ops)
    }

    pub fn new(
        order: usize,
        state: ForwardNodeState<Out>,
        ops: ForwardRecordedOpsBoxed<Out>,
    ) -> Self {
        let id = IdGenerator::generate();
        Self {
            id,
            order,
            state,
            ops,
        }
    }
}
