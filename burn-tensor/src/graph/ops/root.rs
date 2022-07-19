use super::RecordedOps;
use crate::node::{NodeId, NodeStateRef, Ones, Zeros};

#[derive(new, Debug, Clone)]
pub struct InitRecordedOps<Out> {
    root: NodeStateRef<Out>,
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

    fn record(&self, tape: &mut crate::tape::Tape) {
        tape.add(Box::new(self.clone()))
    }
}
