use super::{BackwardRef, RecordedOps};
use crate::node::{NodeStateRef, Ones, Zeros};
use std::ops::Add;

#[derive(new, Debug, Clone)]
pub struct InitRecordedOps {}

impl<Out> RecordedOps<Out> for InitRecordedOps
where
    Out: Clone + Zeros<Out> + Ones<Out> + Add<Output = Out> + 'static,
    Out: std::fmt::Debug,
{
    fn backward(&self, _: &NodeStateRef<Out>) {}
    fn parents(&self) -> Vec<BackwardRef> {
        vec![]
    }
}
