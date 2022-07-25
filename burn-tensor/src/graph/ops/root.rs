use super::{BackwardRecordedOps, ForwardRecordedOps, RecordedOpsParentRef};
use crate::node::{BackwardNodeState, Zeros};
use std::{ops::Add, sync::Arc};

#[derive(new, Debug, Clone)]
pub struct InitRecordedOps {}

impl<Out> BackwardRecordedOps<Out> for InitRecordedOps
where
    Out: Clone + Zeros<Out> + Add<Output = Out> + std::fmt::Debug + 'static,
{
    fn backward_step(&self, _: &BackwardNodeState<Out>) {}
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        vec![]
    }
}

impl<Out> ForwardRecordedOps<Out> for InitRecordedOps
where
    Out: Clone + Zeros<Out> + Add<Output = Out> + std::fmt::Debug + 'static,
{
    fn as_backward(
        &self,
        _graph: &mut super::Forward2BackwardGraphConverter,
    ) -> super::BackwardRecordedOpsRef<Out> {
        Arc::new(self.clone())
    }
}
