use super::NodeStateRef;
use crate::{
    grad::Gradients,
    ops::{RecordedOpsParent, RecordedOpsParentRef, RecordedOpsRef},
};
use std::{collections::HashMap, ops::Add, rc::Rc};

#[derive(Debug)]
pub struct Node<Out> {
    pub id: String,
    pub order: usize,
    pub state: NodeStateRef<Out>,
    pub ops: RecordedOpsRef<Out>,
}

impl<Out> Node<Out> {
    pub fn from_root(state: NodeStateRef<Out>, ops: RecordedOpsRef<Out>) -> Self {
        let order = 0;
        Self::new(order, state, ops)
    }

    pub fn from_unary<T>(
        node: &Node<T>,
        state: NodeStateRef<Out>,
        ops: RecordedOpsRef<Out>,
    ) -> Self {
        let order = node.order + 1;
        Self::new(order, state, ops)
    }
    pub fn from_binary<Lhs, Rhs>(
        lhs: &Node<Lhs>,
        rhs: &Node<Rhs>,
        state: NodeStateRef<Out>,
        ops: RecordedOpsRef<Out>,
    ) -> Self {
        let order = usize::max(lhs.order, rhs.order) + 1;
        Self::new(order, state, ops)
    }

    fn new(order: usize, state: NodeStateRef<Out>, ops: RecordedOpsRef<Out>) -> Self {
        let id = nanoid::nanoid!();
        Self {
            id,
            order,
            state,
            ops,
        }
    }
}

impl<Out> Node<Out>
where
    Out: Zeros<Out> + Ones<Out> + Clone + Add<Output = Out>,
    Out: std::fmt::Debug + 'static,
{
    pub fn backward(&self) -> Gradients {
        let grad = self.state.borrow().value().ones();
        self.state.borrow_mut().update_grad(grad);
        self.ops.backward_step(&self.state);

        let mut nodes = HashMap::with_capacity(self.order);
        let mut parents = self.ops.backward_parents();

        loop {
            match parents.pop() {
                Some(node) => {
                    let order = node.order();

                    if order == 0 {
                        continue;
                    }

                    if nodes.contains_key(&order) {
                        continue;
                    }

                    for parent in node.backward_parents() {
                        if !nodes.contains_key(&parent.order()) {
                            parents.push(parent);
                        }
                    }
                    nodes.insert(order, node);
                }
                None => break,
            }
        }

        for i in (0..self.order + 1).rev() {
            if let Some(node) = nodes.get(&i) {
                node.backward_step();
            }
        }

        Gradients::from(&self)
    }
}

impl<T> RecordedOpsParent for Node<T>
where
    T: Zeros<T> + Clone + Add<Output = T>,
    T: std::fmt::Debug + 'static,
{
    fn backward_step(&self) {
        println!("backward node {}", self.order);
        self.ops.backward_step(&self.state)
    }
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        self.ops.backward_parents()
    }

    fn order(&self) -> usize {
        self.order
    }
    fn id(&self) -> &String {
        &self.id
    }
    fn register_grad(&self, grads: &mut Gradients) {
        grads.register(&self)
    }
}

pub type NodeRef<Out> = Rc<Node<Out>>;

pub trait Zeros<T> {
    fn zeros(&self) -> T;
}
pub trait Ones<T> {
    fn ones(&self) -> T;
}
