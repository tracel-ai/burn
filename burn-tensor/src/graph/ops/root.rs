use super::{RecordedOps, RecordedOpsRef};
use crate::node::{NodeStateRef, Ones, Zeros};
use std::ops::Add;

#[derive(new, Debug, Clone)]
pub struct InitRecordedOps<Out> {
    root: NodeStateRef<Out>,
}

impl<Out> RecordedOps for InitRecordedOps<Out>
where
    Out: Clone + Zeros<Out> + Ones<Out> + Add<Output = Out> + 'static,
    Out: std::fmt::Debug,
{
    fn backward(&self) {}
    fn set_last_ops(&self) {
        let value = self.root.borrow().value();
        self.root.borrow_mut().update_grad(value.ones());
    }

    fn parents_ops(&self) -> Vec<RecordedOpsRef> {
        vec![]
    }
}
