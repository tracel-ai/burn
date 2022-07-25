use super::{BackwardNodeState, BackwardNodeStateRef, ForwardNodeState};
use crate::{
    grad::Gradients,
    ops::{
        BackwardRecordedOpsRef, Forward2BackwardGraphConverter, ForwardRecordedOpsRef,
        RecordedOpsParent, RecordedOpsParentRef,
    },
};
use std::{collections::HashMap, ops::Add, rc::Rc};

#[derive(Debug)]
pub struct Node<Out> {
    pub id: String,
    pub order: usize,
    pub state: ForwardNodeState<Out>,
    pub ops: ForwardRecordedOpsRef<Out>,
}

#[derive(Debug)]
pub struct BackwardNode<Out> {
    pub id: String,
    pub order: usize,
    pub state: BackwardNodeStateRef<Out>,
    pub ops: BackwardRecordedOpsRef<Out>,
}

impl<Out: Clone> BackwardNode<Out> {
    pub fn from_node(node: &NodeRef<Out>, converter: &mut Forward2BackwardGraphConverter) -> Self {
        BackwardNode {
            id: node.id.clone(),
            order: node.order,
            state: BackwardNodeState::new_mut(node.state.value()),
            ops: node.ops.as_backward(converter),
        }
    }
}

impl<Out> Node<Out> {
    pub fn from_root(state: ForwardNodeState<Out>, ops: ForwardRecordedOpsRef<Out>) -> Self {
        let order = 0;
        Self::new(order, state, ops)
    }

    pub fn from_unary<T>(
        node: &Node<T>,
        state: ForwardNodeState<Out>,
        ops: ForwardRecordedOpsRef<Out>,
    ) -> Self {
        let order = node.order + 1;
        Self::new(order, state, ops)
    }
    pub fn from_binary<Lhs, Rhs>(
        lhs: &Node<Lhs>,
        rhs: &Node<Rhs>,
        state: ForwardNodeState<Out>,
        ops: ForwardRecordedOpsRef<Out>,
    ) -> Self {
        let order = usize::max(lhs.order, rhs.order) + 1;
        Self::new(order, state, ops)
    }

    fn new(order: usize, state: ForwardNodeState<Out>, ops: ForwardRecordedOpsRef<Out>) -> Self {
        let id = nanoid::nanoid!();
        Self {
            id,
            order,
            state,
            ops,
        }
    }
}

impl<Out> BackwardNode<Out>
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

impl<T> RecordedOpsParent for BackwardNode<T>
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
pub type BackwardNodeRef<Out> = Rc<BackwardNode<Out>>;

pub trait Zeros<T> {
    fn zeros(&self) -> T;
}
pub trait Ones<T> {
    fn ones(&self) -> T;
}
