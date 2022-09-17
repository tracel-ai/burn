use super::{
    BackwardRecordedOps, BackwardRecordedOpsRef, BinaryOpsNodeState, ForwardRecordedOps,
    RecordedOpsParentRef,
};
use crate::{
    graph::{
        converter::Forward2BackwardGraphConverter,
        node::{BackwardNodeRef, BackwardNodeState, ForwardNodeRef},
    },
    tensor::ops::Zeros,
};
use std::{ops::Add, sync::Arc};

pub trait BinaryOps<Lhs, Rhs, Out>: std::fmt::Debug + Send + Sync {
    fn partial_left(&self, state: &BinaryOpsNodeState<Lhs, Rhs, Out>) -> Lhs;
    fn partial_right(&self, state: &BinaryOpsNodeState<Lhs, Rhs, Out>) -> Rhs;
}

#[derive(new, Debug)]
pub struct ForwardBinaryRecordedOps<Lhs, Rhs, Ops> {
    lhs: ForwardNodeRef<Lhs>,
    rhs: ForwardNodeRef<Rhs>,
    ops: Arc<Ops>,
}

#[derive(new, Debug)]
pub struct BackwardBinaryRecordedOps<Lhs, Rhs, Ops> {
    lhs: BackwardNodeRef<Lhs>,
    rhs: BackwardNodeRef<Rhs>,
    ops: Arc<Ops>,
}

impl<Lhs, Rhs, Out, Ops> ForwardRecordedOps<Out> for ForwardBinaryRecordedOps<Lhs, Rhs, Ops>
where
    Lhs: Clone + Zeros<Lhs> + Add<Output = Lhs> + std::fmt::Debug + 'static + Send + Sync,
    Rhs: Clone + Zeros<Rhs> + Add<Output = Rhs> + std::fmt::Debug + 'static + Send + Sync,
    Out: Clone + Zeros<Out> + Add<Output = Out> + std::fmt::Debug + 'static,
    Ops: BinaryOps<Lhs, Rhs, Out> + std::fmt::Debug + 'static + Send + Sync,
{
    fn to_backward(
        &self,
        graph: &mut Forward2BackwardGraphConverter,
    ) -> BackwardRecordedOpsRef<Out> {
        let lhs = graph.from(&self.lhs);
        let rhs = graph.from(&self.rhs);
        let ops = self.ops.clone();

        Arc::new(BackwardBinaryRecordedOps::new(lhs, rhs, ops))
    }
}

impl<Lhs, Rhs, Out, Ops> BackwardRecordedOps<Out> for BackwardBinaryRecordedOps<Lhs, Rhs, Ops>
where
    Lhs: Clone + Zeros<Lhs> + Add<Output = Lhs> + std::fmt::Debug + 'static + Send + Sync,
    Rhs: Clone + Zeros<Rhs> + Add<Output = Rhs> + std::fmt::Debug + 'static + Send + Sync,
    Out: Clone + Zeros<Out> + Add<Output = Out> + std::fmt::Debug + 'static,
    Ops: BinaryOps<Lhs, Rhs, Out> + std::fmt::Debug + 'static,
{
    fn backward_step(&self, state: &BackwardNodeState<Out>) {
        let state = BinaryOpsNodeState::new(&self.lhs.state, &self.rhs.state, state);

        let partial_left: Lhs = self.ops.partial_left(&state);
        let partial_right: Rhs = self.ops.partial_right(&state);

        self.lhs.state.update_grad(partial_left);
        self.rhs.state.update_grad(partial_right);
    }

    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }
}
