use super::NodeStateRef;
use crate::{ops::RecordedOpsRef, tape::Tape};
use std::rc::Rc;

#[derive(new, Debug)]
pub struct Node<Out> {
    pub state: NodeStateRef<Out>,
    pub ops: RecordedOpsRef,
}

impl<Out> Node<Out> {
    pub fn record(&self, tape: &mut Tape) {
        let mut all_ops = self.ops.parents_ops();
        tape.add(self.ops.clone());

        loop {
            if all_ops.len() == 0 {
                return;
            }
            let ops = all_ops.pop().unwrap();
            all_ops.append(&mut ops.parents_ops());
            tape.add(ops);
        }
    }
}
pub type NodeRef<Out> = Rc<Node<Out>>;

pub trait Zeros<T> {
    fn zeros(&self) -> T;
}
pub trait Ones<T> {
    fn ones(&self) -> T;
}
