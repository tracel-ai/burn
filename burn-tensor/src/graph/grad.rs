use crate::{
    graph::{
        node::{BackwardNode, ForwardNode},
        traversal::{BreadthFirstSearch, GraphTraversal},
    },
    tensor::ops::Zeros,
};
use std::{any::Any, collections::HashMap, ops::Add};

#[derive(Default)]
pub struct Gradients {
    grads: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl Gradients {
    pub fn empty() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    pub fn register<T>(&mut self, node: &BackwardNode<T>)
    where
        T: Zeros + Clone + Add<Output = T>,
        T: std::fmt::Debug + 'static + Send + Sync,
    {
        let grad = node.state.grad();
        self.grads.insert(node.id.clone(), Box::new(grad));
    }

    pub fn register_any<V>(&mut self, id: String, value: V)
    where
        V: std::fmt::Debug + 'static + Send + Sync,
    {
        self.grads.insert(id, Box::new(value));
    }

    pub fn from<T>(node: &BackwardNode<T>) -> Self
    where
        T: Zeros + Clone + Add<Output = T>,
        T: std::fmt::Debug + 'static + Send + Sync,
    {
        let mut grads = Self::empty();
        let traversal = BreadthFirstSearch::new(node);
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

    pub fn get<V: 'static>(&self, id: &str) -> Option<&V> {
        let grad = match self.grads.get(id) {
            Some(grad) => grad,
            None => return None,
        };

        grad.downcast_ref()
    }
}

pub trait AsNode<T> {
    fn as_node(&self) -> &ForwardNode<T>;
}
