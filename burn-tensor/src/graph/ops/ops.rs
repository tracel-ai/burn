use crate::{
    grad::Gradients,
    node::{BackwardNode, BackwardNodeRef, BackwardNodeStateRef, ForwardNodeRef},
};
use std::{any::Any, collections::HashMap, rc::Rc};

#[derive(new)]
pub struct BinaryOpsNodeState<'a, Lhs, Rhs, Out> {
    pub left: &'a BackwardNodeStateRef<Lhs>,
    pub right: &'a BackwardNodeStateRef<Rhs>,
    pub output: &'a BackwardNodeStateRef<Out>,
}

#[derive(new)]
pub struct UnaryOpsNodeState<'a, In, Out> {
    pub input: &'a BackwardNodeStateRef<In>,
    pub output: &'a BackwardNodeStateRef<Out>,
}

pub trait BackwardRecordedOps<T>: std::fmt::Debug {
    fn backward_step(&self, state: &BackwardNodeStateRef<T>);
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef>;
}

pub struct Forward2BackwardGraphConverter {
    state: HashMap<String, Box<dyn Any>>,
}

impl Forward2BackwardGraphConverter {
    pub fn empty() -> Self {
        Self {
            state: HashMap::new(),
        }
    }
    pub fn from<T: Clone + 'static>(&mut self, node: &ForwardNodeRef<T>) -> BackwardNodeRef<T> {
        match self.state.get(&node.id) {
            Some(node) => {
                let node: &BackwardNodeRef<T> = node.downcast_ref().unwrap();
                return node.clone();
            }
            None => {}
        };

        let node = Rc::new(BackwardNode::from_node(node, self));
        self.state.insert(node.id.clone(), Box::new(node.clone()));
        node
    }
}
pub trait ForwardRecordedOps<T>: std::fmt::Debug {
    fn as_backward(&self, graph: &mut Forward2BackwardGraphConverter) -> BackwardRecordedOpsRef<T>;
}

pub trait RecordedOpsParent: std::fmt::Debug {
    fn order(&self) -> usize;
    fn id(&self) -> &String;
    fn backward_step(&self);
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef>;
    fn register_grad(&self, grads: &mut Gradients);
}

pub type ForwardRecordedOpsRef<T> = Rc<dyn ForwardRecordedOps<T>>;
pub type BackwardRecordedOpsRef<T> = Rc<dyn BackwardRecordedOps<T>>;
pub type RecordedOpsParentRef = Rc<dyn RecordedOpsParent>;
