use crate::ops::RecordedOpsParentRef;

#[derive(Debug)]
pub struct Tape {
    pub operations: Vec<RecordedOpsParentRef>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn backward(&mut self) {
        for ops in self.operations.iter_mut() {
            ops.backward_step();
        }
    }

    pub fn add(&mut self, ops: RecordedOpsParentRef) {
        self.operations.push(ops)
    }
}
