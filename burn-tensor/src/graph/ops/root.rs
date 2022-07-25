use super::{BackwardRecordedOps, ForwardRecordedOps, RecordedOpsParentRef};
use crate::node::{BackwardNodeStateRef, Zeros};
use std::{ops::Add, rc::Rc};

#[derive(new, Debug, Clone)]
pub struct InitRecordedOps {}

impl<Out> BackwardRecordedOps<Out> for InitRecordedOps
where
    Out: Clone + Zeros<Out> + Add<Output = Out> + std::fmt::Debug + 'static,
{
    fn backward_step(&self, _: &BackwardNodeStateRef<Out>) {}
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
        Rc::new(self.clone())
    }
}
