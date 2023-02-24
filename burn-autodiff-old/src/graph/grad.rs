use crate::graph::{
    node::{BackwardNode, ForwardNode},
    traversal::{BreadthFirstSearch, GraphTraversal},
};
use burn_tensor::ops::Zeros;
use std::{any::Any, collections::HashMap, ops::Add};

#[derive(Default, Debug)]
pub struct Grads {
    grads: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl Grads {
    pub fn register_node<T>(&mut self, node: &BackwardNode<T>)
    where
        T: Zeros + Clone + Add<Output = T>,
        T: std::fmt::Debug + 'static + Send + Sync,
    {
        let grad = node.state.grad();
        self.grads.insert(node.id.clone(), Box::new(grad));
    }

    pub fn from_node<T>(node: &BackwardNode<T>) -> Self
    where
        T: Zeros + Clone + Add<Output = T>,
        T: std::fmt::Debug + 'static + Send + Sync,
    {
        let mut grads = Self::default();
        let traversal = BreadthFirstSearch::new(node);
        grads.register_node(node);

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

pub trait AsNode<T> {
    fn as_node(&self) -> &ForwardNode<T>;
}
