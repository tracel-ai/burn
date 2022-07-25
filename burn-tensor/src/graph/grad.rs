use crate::node::{BackwardNode, ForwardNode, Zeros};
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    ops::Add,
};

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
        let mut visited = HashSet::new();
        let mut parents = node.ops.backward_parents();

        visited.insert(node.id.clone());
        grads.register(node);

        loop {
            match parents.pop() {
                Some(node) => {
                    let id = node.id();

                    if visited.contains(id) {
                        continue;
                    }

                    visited.insert(id.clone());
                    node.register_grad(&mut grads);

                    for ops in node.backward_parents() {
                        if !visited.contains(ops.id()) {
                            parents.push(ops);
                        }
                    }
                }
                None => break,
            }
        }

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
