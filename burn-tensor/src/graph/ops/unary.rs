use super::{RecordedOps, RecordedOpsParentRef, UnaryOpsNodeState};
use crate::node::{NodeRef, NodeStateRef, Zeros};
use std::ops::Add;

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
    In: Clone + Zeros<In> + Add<Output = In> + std::fmt::Debug + 'static,
    Out: Clone + Zeros<Out> + Add<Output = Out> + std::fmt::Debug + 'static,
    Ops: UnaryOps<In, Out> + std::fmt::Debug + 'static,
{
    fn backward_step(&self, state: &NodeStateRef<Out>) {
        let input = self.input.state.borrow().value();
        let output = state.borrow().value();
        let state = UnaryOpsNodeState::new(&input, &output);

        let partial = self.ops.partial(&state);
        self.input.state.borrow_mut().update_grad(partial);
    }
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        vec![self.input.clone()]
    }
}
