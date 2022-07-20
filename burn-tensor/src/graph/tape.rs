use crate::ops::RecordedOpsRef;

#[derive(Debug)]
pub struct Tape {
    pub operations: Vec<RecordedOpsRef>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn backward(&mut self) {
        for ops in self.operations.iter_mut() {
            ops.backward();
        }
    }

    pub fn add(&mut self, ops: RecordedOpsRef) {
        self.operations.push(ops)
    }
}
