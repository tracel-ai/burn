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

pub trait RecordedOps: std::fmt::Debug {
    fn backward(&self);
    fn set_last_ops(&self);
    fn parents_ops(&self) -> Vec<ParentOps>;
}
pub type RecordedOpsRef = Rc<dyn RecordedOps>;

pub struct ParentOps {
    pub id: usize,
    pub ops: RecordedOpsRef,
}

impl ParentOps {
    pub fn from<T>(node: &Node<T>) -> Self {
        Self {
            id: node.id.clone(),
            ops: node.ops.clone(),
        }
    }
}
