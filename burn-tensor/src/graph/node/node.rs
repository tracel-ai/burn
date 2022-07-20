use super::NodeStateRef;
use crate::ops::{RecordedOpsParent, RecordedOpsParentRef, RecordedOpsRef};
use std::{ops::Add, rc::Rc};

#[derive(Debug)]
pub struct Node<Out> {
    pub state: NodeStateRef<Out>,
    pub ops: RecordedOpsRef<Out>,
}

impl<Out> Node<Out> {
    pub fn new(state: NodeStateRef<Out>, ops: RecordedOpsRef<Out>) -> Self {
        Self { state, ops }
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
        let mut parents = self.ops.backward_parents();

        loop {
            if let Some(node) = parents.pop() {
                node.backward_step();

                for parent in node.backward_parents() {
                    parents.push(parent);
                }
            } else {
                break;
            }
        }
    }
}

impl<T: std::fmt::Debug> RecordedOpsParent for Node<T> {
    fn backward_step(&self) {
        self.ops.backward_step(&self.state)
    }
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        self.ops.backward_parents()
    }
}

pub type NodeRef<Out> = Rc<Node<Out>>;

pub trait Zeros<T> {
    fn zeros(&self) -> T;
}
pub trait Ones<T> {
    fn ones(&self) -> T;
}
