use super::{BackwardRecordedOps, ForwardRecordedOps, RecordedOpsParentRef, UnaryOpsNodeState};
use crate::{
    graph::{
        converter::Forward2BackwardGraphConverter,
        node::{BackwardNodeRef, BackwardNodeState, ForwardNodeRef},
    },
    tensor::ops::Zeros,
};
use std::{ops::Add, sync::Arc};

pub trait UnaryOps<In, Out>: std::fmt::Debug + Send + Sync {
    fn partial(&self, state: &UnaryOpsNodeState<In, Out>) -> In;
}

#[derive(new, Debug)]
pub struct ForwardUnaryRecordedOps<In, Ops> {
    input: ForwardNodeRef<In>,
    ops: Arc<Ops>,
}

#[derive(new, Debug)]
pub struct BackwareUnaryRecordedOps<In, Ops> {
    input: BackwardNodeRef<In>,
    ops: Arc<Ops>,
}

impl<In, Out, Ops> ForwardRecordedOps<Out> for ForwardUnaryRecordedOps<In, Ops>
where
    In: Clone + Zeros + Add<Output = In> + std::fmt::Debug + 'static + Send + Sync,
    Out: Clone + Zeros + Add<Output = Out> + std::fmt::Debug + 'static,
    Ops: UnaryOps<In, Out> + std::fmt::Debug + 'static,
{
    fn to_backward(
        &self,
        graph: &mut Forward2BackwardGraphConverter,
    ) -> super::BackwardRecordedOpsRef<Out> {
        let input = graph.from(&self.input);
        let ops = self.ops.clone();

        Arc::new(BackwareUnaryRecordedOps::new(input, ops))
    }
}

impl<In, Out, Ops> BackwardRecordedOps<Out> for BackwareUnaryRecordedOps<In, Ops>
where
    In: Clone + Zeros + Add<Output = In> + std::fmt::Debug + 'static + Send + Sync,
    Out: Clone + Zeros + Add<Output = Out> + std::fmt::Debug + 'static,
    Ops: UnaryOps<In, Out> + std::fmt::Debug + 'static,
{
    fn backward_step(&self, state: &BackwardNodeState<Out>) {
        let state = UnaryOpsNodeState::new(&self.input.state, state);
        let partial = self.ops.partial(&state);
        self.input.state.update_grad(partial);
    }
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        vec![self.input.clone()]
    }
}
