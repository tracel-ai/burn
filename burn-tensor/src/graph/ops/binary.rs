use super::{BinaryOpsNodeState, RecordedOps, RecordedOpsParentRef};
use crate::node::{NodeRef, NodeStateRef, Ones, Zeros};
use std::ops::{Add, Mul};

pub trait BinaryOps<Lhs, Rhs, Out>: std::fmt::Debug {
    fn partial_left(&self, state: &BinaryOpsNodeState<Lhs, Rhs, Out>) -> Lhs;
    fn partial_right(&self, state: &BinaryOpsNodeState<Lhs, Rhs, Out>) -> Rhs;
}

#[derive(new, Debug)]
pub struct BinaryRecordedOps<Lhs, Rhs, Ops> {
    lhs: NodeRef<Lhs>,
    rhs: NodeRef<Rhs>,
    ops: Ops,
}

impl<Lhs, Rhs, Out, Ops> RecordedOps<Out> for BinaryRecordedOps<Lhs, Rhs, Ops>
where
    Lhs: Clone
        + Zeros<Lhs>
        + Mul<Out, Output = Lhs>
        + Add<Output = Lhs>
        + Add<Out, Output = Lhs>
        + 'static,
    Rhs: Clone
        + Zeros<Rhs>
        + Mul<Out, Output = Rhs>
        + Add<Output = Rhs>
        + Add<Out, Output = Rhs>
        + 'static,
    Out: Clone + Zeros<Out> + Ones<Out> + Add<Output = Out> + 'static,
    Lhs: std::fmt::Debug,
    Rhs: std::fmt::Debug,
    Out: std::fmt::Debug,
    Ops: BinaryOps<Lhs, Rhs, Out> + 'static,
{
    fn backward_step(&self, state: &NodeStateRef<Out>) {
        let state = BinaryOpsNodeState::new(&self.lhs.state, &self.rhs.state, state);

        let partial_left: Lhs = self.ops.partial_left(&state);
        let partial_right: Rhs = self.ops.partial_right(&state);

        self.lhs.state.borrow_mut().update_grad(partial_left);
        self.rhs.state.borrow_mut().update_grad(partial_right);
    }

    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }
}
