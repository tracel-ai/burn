use super::NodeStateRef;
use crate::ops::{RecordedOpsParent, RecordedOpsParentRef, RecordedOpsRef};
use std::{collections::HashMap, ops::Add, rc::Rc};

#[derive(Debug)]
pub struct Node<Out> {
    pub order: usize,
    pub state: NodeStateRef<Out>,
    pub ops: RecordedOpsRef<Out>,
}

impl<Out> Node<Out> {
    pub fn from_root(state: NodeStateRef<Out>, ops: RecordedOpsRef<Out>) -> Self {
        let order = 0;
        Self { order, state, ops }
    }

    pub fn from_unary<T>(
        node: &Node<T>,
        state: NodeStateRef<Out>,
        ops: RecordedOpsRef<Out>,
    ) -> Self {
        let order = node.order + 1;
        Self { order, state, ops }
    }
    pub fn from_binary<Lhs, Rhs>(
        lhs: &Node<Lhs>,
        rhs: &Node<Rhs>,
        state: NodeStateRef<Out>,
        ops: RecordedOpsRef<Out>,
    ) -> Self {
        let order = usize::max(lhs.order, rhs.order) + 1;
        Self { order, state, ops }
    }
}

impl<Out> Node<Out>
where
    Out: Zeros<Out> + Ones<Out> + Clone + Add<Output = Out>,
    Out: std::fmt::Debug,
{
    pub fn backward(&self) {
        let grad = self.state.borrow().value().ones();
        self.state.borrow_mut().update_grad(grad);
        self.ops.backward_step(&self.state);

        let mut nodes = HashMap::new();
        let mut parents = self.ops.backward_parents();

        loop {
            match parents.pop() {
                Some(node) => {
                    let id = node.id();

                    if id == 0 {
                        continue;
                    }

                    if nodes.contains_key(&id) {
                        continue;
                    }

                    for parent in node.backward_parents() {
                        if !nodes.contains_key(&parent.id()) {
                            parents.push(parent);
                        }
                    }
                    nodes.insert(id, node);
                }
                None => break,
            }
        }

        for i in (0..self.order + 1).rev() {
            if let Some(node) = nodes.get(&i) {
                node.backward_step();
            }
        }
    }
}

impl<T: std::fmt::Debug> RecordedOpsParent for Node<T> {
    fn backward_step(&self) {
        println!("backward node {}", self.order);
        self.ops.backward_step(&self.state)
    }
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        self.ops.backward_parents()
    }

    fn id(&self) -> usize {
        self.order
    }
}

pub type NodeRef<Out> = Rc<Node<Out>>;

pub trait Zeros<T> {
    fn zeros(&self) -> T;
}
pub trait Ones<T> {
    fn ones(&self) -> T;
}
