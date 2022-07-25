use crate::{grad::Gradients, node::NodeStateRef};
use std::rc::Rc;

#[derive(new)]
pub struct BinaryOpsNodeState<'a, Lhs, Rhs, Out> {
    pub left: &'a NodeStateRef<Lhs>,
    pub right: &'a NodeStateRef<Rhs>,
    pub output: &'a NodeStateRef<Out>,
}

#[derive(new)]
pub struct UnaryOpsNodeState<'a, In, Out> {
    pub input: &'a NodeStateRef<In>,
    pub output: &'a NodeStateRef<Out>,
}

pub trait RecordedOps<T>: std::fmt::Debug {
    fn backward_step(&self, state: &NodeStateRef<T>);
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef>;
}

pub trait RecordedOpsParent: std::fmt::Debug {
    fn order(&self) -> usize;
    fn id(&self) -> &String;
    fn backward_step(&self);
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef>;
    fn register_grad(&self, grads: &mut Gradients);
}

pub type RecordedOpsRef<T> = Rc<dyn RecordedOps<T>>;
pub type RecordedOpsParentRef = Rc<dyn RecordedOpsParent>;
