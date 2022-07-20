use crate::ops::BackwardRef;

#[derive(Debug)]
pub struct Tape {
    pub operations: Vec<BackwardRef>,
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

    pub fn add(&mut self, ops: BackwardRef) {
        self.operations.push(ops)
    }
}
