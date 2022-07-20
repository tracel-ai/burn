use super::ADTensor;
use crate::{node::Zeros, tape::Tape};
use std::ops::Add;

impl<T, P, const D: usize> ADTensor<P, D, T> {
    pub fn backprob(&self) {
        let mut tape = Tape::new();
        self.node.ops.set_last_ops();
        self.node.record(&mut tape);
        tape.backward();
    }
}

impl<T, P, const D: usize> ADTensor<P, D, T>
where
    T: Zeros<T> + Clone + Add<Output = T>,
    T: std::fmt::Debug,
{
    pub fn grad(&self) -> T {
        self.node.state.borrow_mut().grad()
    }
}
