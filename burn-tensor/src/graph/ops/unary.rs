use super::{BackwardRef, RecordedOps, UnaryOpsNodeState};
use crate::node::{NodeRef, NodeStateRef, Ones, Zeros};
use std::ops::{Add, Mul};

pub trait UnaryOps<In, Out>: std::fmt::Debug {
    fn partial(&self, state: &UnaryOpsNodeState<In, Out>) -> In;
}

#[derive(new, Debug)]
pub struct UnaryRecordedOps<In, Ops> {
    input: NodeRef<In>,
    ops: Ops,
}

impl<In, Out, Ops> RecordedOps<Out> for UnaryRecordedOps<In, Ops>
where
    In: Clone + Zeros<In> + Mul<Out, Output = In> + Add<Output = In> + 'static,
    Out: Clone + Zeros<Out> + Ones<Out> + Add<Output = Out> + 'static,
    In: std::fmt::Debug,
    Out: std::fmt::Debug,
    Ops: UnaryOps<In, Out> + 'static,
{
    fn backward(&self, state: &NodeStateRef<Out>) {
        let input = self.input.state.borrow().value();
        let output = state.borrow().value();
        let state = UnaryOpsNodeState::new(&input, &output);

        let partial = self.ops.partial(&state);
        self.input.state.borrow_mut().update_grad(partial);
    }
    fn parents(&self) -> Vec<BackwardRef> {
        vec![self.input.clone()]
    }
}
