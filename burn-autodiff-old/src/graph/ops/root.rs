use super::{BackwardRecordedOps, ForwardRecordedOps, RecordedOpsParentRef};
use crate::graph::{converter::Forward2BackwardGraphConverter, node::BackwardNodeState};
use burn_tensor::ops::Zeros;
use std::ops::Add;

#[derive(new, Debug, Clone)]
pub struct InitRecordedOps {}

impl<Out> BackwardRecordedOps<Out> for InitRecordedOps
where
    Out: Clone + Zeros + Add<Output = Out> + std::fmt::Debug + 'static,
{
    fn backward_step(&self, _: &BackwardNodeState<Out>) {}
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        vec![]
    }
}

impl<Out> ForwardRecordedOps<Out> for InitRecordedOps
where
    Out: Clone + Zeros + Add<Output = Out> + std::fmt::Debug + 'static,
{
    fn to_backward(
        &self,
        _graph: &mut Forward2BackwardGraphConverter,
    ) -> super::BackwardRecordedOpsBoxed<Out> {
        Box::new(self.clone())
    }
}
