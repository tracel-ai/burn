use crate::graph::{
    node::{BackwardNode, ForwardNode},
    traversal::{BreadthFirstSearch, GraphTraversal},
};
use burn_tensor::{
    backend::{ADBackend, Gradients},
    ops::Zeros,
    Tensor,
};
use std::{any::Any, collections::HashMap, ops::Add};

#[derive(Default)]
pub struct Grads {
    grads: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl<B: ADBackend> Gradients<B> for Grads {
    fn empty() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }
    fn get<const D: usize>(&self, id: &str) -> Option<&Tensor<B::InnerBackend, D>> {
        let grad = match self.grads.get(id) {
            Some(grad) => grad,
            None => return None,
        };

        grad.downcast_ref()
    }
    fn register<const D: usize>(&mut self, id: String, value: Tensor<B::InnerBackend, D>) {
        self.grads.insert(id, Box::new(value));
    }

    fn merge(mut grads: Vec<Self>) -> Self
    where
        Self: Sized,
    {
        if grads.len() <= 0 {
            return <Self as Gradients<B>>::empty();
        }

        let mut grads_merged = grads.pop().unwrap();

        for grad_item in grads {
            for (key, item) in grad_item.grads {
                if let Some(item_other) = grads_merged.grads.remove(&key) {}
            }
        }

        grads_merged
    }
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
