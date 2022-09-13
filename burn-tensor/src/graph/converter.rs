use crate::{
    graph::node::{BackwardNode, BackwardNodeRef, ForwardNodeRef},
    tensor::ops::Zeros,
};
use std::{any::Any, collections::HashMap, sync::Arc};

#[derive(Default)]
pub struct Forward2BackwardGraphConverter {
    state: HashMap<String, Box<dyn Any>>,
}

impl Forward2BackwardGraphConverter {
    pub fn new() -> Self {
        Self {
            state: HashMap::new(),
        }
    }
    pub fn from<T: Clone + 'static + Zeros<T>>(
        &mut self,
        node: &ForwardNodeRef<T>,
    ) -> BackwardNodeRef<T> {
        if let Some(node) = self.state.get(&node.id) {
            let node: &BackwardNodeRef<T> = node.downcast_ref().unwrap();
            return node.clone();
        };

        let node = Arc::new(BackwardNode::from_node(node, self));
        self.state.insert(node.id.clone(), Box::new(node.clone()));
        node
    }
}
