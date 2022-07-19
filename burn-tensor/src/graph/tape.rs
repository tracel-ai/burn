use crate::{node::NodeId, ops::RecordedOpsRef};
use std::{cell::RefCell, rc::Rc};

#[derive(Debug)]
pub struct Tape {
    pub operations: Vec<RecordedOpsRef>,
}
pub type TapeRef = Rc<RefCell<Tape>>;

impl Tape {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn new_ref() -> TapeRef {
        Rc::new(RefCell::new(Self::new()))
    }

    pub fn backward(&mut self, from: NodeId) {
        let mut init = false;

        for ops in self.operations.iter_mut().rev() {
            if init {
                ops.backward();
            } else if ops.id() == from {
                init = true;
                ops.set_last_ops();
                ops.backward();
            }
        }
    }

    pub fn add(&mut self, ops: RecordedOpsRef) {
        self.operations.push(ops)
    }
}
