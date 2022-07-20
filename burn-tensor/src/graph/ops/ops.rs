use crate::node::{Node, NodeStateRef};
use std::rc::Rc;

#[derive(new)]
pub struct BinaryOpsNodeState<'a, Lhs, Rhs, Out> {
    pub left: &'a NodeStateRef<Lhs>,
    pub right: &'a NodeStateRef<Rhs>,
    pub output: &'a NodeStateRef<Out>,
}

#[derive(new)]
pub struct UnaryOpsNodeState<'a, In, Out> {
    pub input: &'a In,
    pub output: &'a Out,
}

pub trait RecordedOps<T>: std::fmt::Debug {
    fn backward(&self, state: &NodeStateRef<T>);
    fn parents(&self) -> Vec<BackwardRef>;
}
pub type RecordedOpsRef<T> = Rc<dyn RecordedOps<T>>;

pub trait Backward: std::fmt::Debug {
    fn backward(&self);
    fn parents(&self) -> Vec<BackwardRef>;
}
pub type BackwardRef = Rc<dyn Backward>;

impl<T: std::fmt::Debug> Backward for Node<T> {
    fn backward(&self) {
        self.ops.backward(&self.state)
    }
    fn parents(&self) -> Vec<BackwardRef> {
        self.ops.parents()
    }
}
