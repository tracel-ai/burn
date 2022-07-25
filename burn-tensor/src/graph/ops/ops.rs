use crate::{
    grad::Gradients,
    node::{BackwardNode, BackwardNodeRef, BackwardNodeState, ForwardNodeRef, Zeros},
};
use std::{any::Any, collections::HashMap, sync::Arc};

#[derive(new)]
pub struct BinaryOpsNodeState<'a, Lhs, Rhs, Out> {
    pub left: &'a BackwardNodeState<Lhs>,
    pub right: &'a BackwardNodeState<Rhs>,
    pub output: &'a BackwardNodeState<Out>,
}

#[derive(new)]
pub struct UnaryOpsNodeState<'a, In, Out> {
    pub input: &'a BackwardNodeState<In>,
    pub output: &'a BackwardNodeState<Out>,
}

pub trait BackwardRecordedOps<T>: std::fmt::Debug {
    fn backward_step(&self, state: &BackwardNodeState<T>);
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
    pub fn from<T: Clone + 'static + Zeros<T>>(
        &mut self,
        node: &ForwardNodeRef<T>,
    ) -> BackwardNodeRef<T> {
        match self.state.get(&node.id) {
            Some(node) => {
                let node: &BackwardNodeRef<T> = node.downcast_ref().unwrap();
                return node.clone();
            }
            None => {}
        };

        let node = Arc::new(BackwardNode::from_node(node, self));
        self.state.insert(node.id.clone(), Box::new(node.clone()));
        node
    }
}
pub trait ForwardRecordedOps<T>: std::fmt::Debug + Send + Sync {
    fn as_backward(&self, graph: &mut Forward2BackwardGraphConverter) -> BackwardRecordedOpsRef<T>;
}

pub trait RecordedOpsParent: std::fmt::Debug {
    fn order(&self) -> usize;
    fn id(&self) -> &String;
    fn backward_step(&self);
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef>;
    fn register_grad(&self, grads: &mut Gradients);
}

pub type ForwardRecordedOpsRef<T> = Arc<dyn ForwardRecordedOps<T>>;
pub type BackwardRecordedOpsRef<T> = Arc<dyn BackwardRecordedOps<T>>;
pub type RecordedOpsParentRef = Arc<dyn RecordedOpsParent>;
