use super::RecordedOps;
use crate::node::{NodeId, NodeRef, Ones, Zeros};

#[derive(new, Debug)]
pub struct InitRecordedOps<Out> {
    root: NodeRef<Out>,
}

impl<Out> RecordedOps for InitRecordedOps<Out>
where
    Out: Clone + Zeros<Out> + Ones<Out> + 'static,
    Out: std::fmt::Debug,
{
    fn id(&self) -> NodeId {
        self.root.borrow().id()
    }

    fn backward(&mut self) {}
    fn set_last_ops(&mut self) {
        let value = self.root.borrow().value();
        self.root.borrow_mut().update_grad(value.ones());
    }
}
