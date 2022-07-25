use crate::{
    node::{BackwardNode, ForwardNode, Zeros},
    traversal::{BreadthFirstSearch, GraphTraversal},
};
use std::{any::Any, collections::HashMap, ops::Add};

pub struct Gradients {
    grads: HashMap<String, Box<dyn Any>>,
}

impl Gradients {
    pub fn register<T>(&mut self, node: &BackwardNode<T>)
    where
        T: Zeros<T> + Clone + Add<Output = T>,
        T: std::fmt::Debug + 'static,
    {
        let grad = node.state.grad();
        self.grads.insert(node.id.clone(), Box::new(grad));
    }
    fn empty() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }
}

pub trait AsNode<T> {
    fn as_node(&self) -> &ForwardNode<T>;
}

impl Gradients {
    pub fn from<T>(node: &BackwardNode<T>) -> Self
    where
        T: Zeros<T> + Clone + Add<Output = T>,
        T: std::fmt::Debug + 'static,
    {
        let mut grads = Self::empty();
        let traversal = BreadthFirstSearch::new(&node);
        grads.register(node);

        traversal.traverse(|node| {
            node.register_grad(&mut grads);
        });

        grads
    }

    pub fn wrt<T: 'static, V: AsNode<T>>(&self, variable: &V) -> Option<&T> {
        let node = variable.as_node();
        let grad = match self.grads.get(&node.id) {
            Some(grad) => grad,
            None => return None,
        };

        grad.downcast_ref()
    }
}
