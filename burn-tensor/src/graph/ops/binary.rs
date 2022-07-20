use super::{BinaryOpsNodeState, RecordedOps, RecordedOpsRef};
use crate::node::{NodeRef, NodeStateRef, Ones, Zeros};
use std::ops::{Add, Mul};

pub trait BinaryOps<Lhs, Rhs, Out>: std::fmt::Debug {
    fn partial_left(&self, state: &BinaryOpsNodeState<Lhs, Rhs, Out>) -> Lhs;
    fn partial_right(&self, state: &BinaryOpsNodeState<Lhs, Rhs, Out>) -> Rhs;
}

#[derive(new, Debug)]
pub struct BinaryRecordedOps<Lhs, Rhs, Out, Ops> {
    lhs: NodeRef<Lhs>,
    rhs: NodeRef<Rhs>,
    out: NodeStateRef<Out>,
    ops: Ops,
}

impl<Lhs, Rhs, Out, Ops> RecordedOps for BinaryRecordedOps<Lhs, Rhs, Out, Ops>
where
    Lhs: Clone + Zeros<Lhs> + Mul<Out, Output = Lhs> + Add<Output = Lhs> + 'static,
    Rhs: Clone + Zeros<Rhs> + Mul<Out, Output = Rhs> + Add<Output = Rhs> + 'static,
    Out: Clone + Zeros<Out> + Ones<Out> + Add<Output = Out> + 'static,
    Lhs: std::fmt::Debug,
    Rhs: std::fmt::Debug,
    Out: std::fmt::Debug,
    Ops: BinaryOps<Lhs, Rhs, Out> + 'static,
{
    fn backward(&self) {
        let state = BinaryOpsNodeState::new(&self.lhs.state, &self.rhs.state, &self.out);

        let partial_left = self.ops.partial_left(&state);
        let partial_right: Rhs = self.ops.partial_right(&state);

        let grad_mine = self.out.borrow_mut().grad();

        self.lhs
            .state
            .borrow_mut()
            .update_grad(partial_left * grad_mine.clone());
        self.rhs
            .state
            .borrow_mut()
            .update_grad(partial_right * grad_mine);
    }

    fn set_last_ops(&self) {
        let value = self.out.borrow().value();
        self.out.borrow_mut().update_grad(value.ones());
    }

    fn parents_ops(&self) -> Vec<RecordedOpsRef> {
        vec![self.lhs.ops.clone(), self.rhs.ops.clone()]
    }
}
