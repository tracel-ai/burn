use super::{RecordedOps, RecordedOpsParentRef};
use crate::node::{NodeStateRef, Zeros};
use std::ops::Add;

#[derive(new, Debug, Clone)]
pub struct InitRecordedOps {}

impl<Out> RecordedOps<Out> for InitRecordedOps
where
    Out: Clone + Zeros<Out> + Add<Output = Out> + std::fmt::Debug + 'static,
{
    fn backward_step(&self, _: &NodeStateRef<Out>) {}
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        vec![]
    }
}
