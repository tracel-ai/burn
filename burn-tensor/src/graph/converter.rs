use crate::node::{BackwardNode, BackwardNodeRef, ForwardNodeRef, Zeros};
use std::{any::Any, collections::HashMap, sync::Arc};

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
