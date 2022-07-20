use super::{ParentOps, RecordedOps, UnaryOpsNodeState};
use crate::node::{NodeRef, NodeStateRef, Ones, Zeros};
use std::ops::{Add, Mul};

pub trait UnaryOps<In, Out>: std::fmt::Debug {
    fn partial(&self, state: &UnaryOpsNodeState<In, Out>) -> In;
}

#[derive(new, Debug)]
pub struct UnaryRecordedOps<In, Out, Ops> {
    input: NodeRef<In>,
    out: NodeStateRef<Out>,
    ops: Ops,
}

impl<In, Out, Ops> RecordedOps for UnaryRecordedOps<In, Out, Ops>
where
    In: Clone + Zeros<In> + Mul<Out, Output = In> + Add<Output = In> + 'static,
    Out: Clone + Zeros<Out> + Ones<Out> + Add<Output = Out> + 'static,
    In: std::fmt::Debug,
    Out: std::fmt::Debug,
    Ops: UnaryOps<In, Out> + 'static,
{
    fn backward(&self) {
        let input = self.input.state.borrow().value();
        let output = self.out.borrow().value();
        let state = UnaryOpsNodeState::new(&input, &output);

        let partial = self.ops.partial(&state);
        self.input.state.borrow_mut().update_grad(partial);
    }

    fn set_last_ops(&self) {
        let value = self.out.borrow().value();
        self.out.borrow_mut().update_grad(value.ones());
    }

    fn parents_ops(&self) -> Vec<ParentOps> {
        vec![ParentOps::from(&self.input)]
    }
}
