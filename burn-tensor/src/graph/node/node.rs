use super::NodeStateRef;
use crate::ops::{RecordedOpsParent, RecordedOpsParentRef, RecordedOpsRef};
use std::{collections::HashSet, ops::Add, rc::Rc};

#[derive(Debug)]
pub struct Node<Out> {
    pub id: String,
    pub state: NodeStateRef<Out>,
    pub ops: RecordedOpsRef<Out>,
}

impl<Out> Node<Out> {
    pub fn new(state: NodeStateRef<Out>, ops: RecordedOpsRef<Out>) -> Self {
        let id = nanoid::nanoid!();
        println!("Creating node {}", id);
        Self { id, state, ops }
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
        let mut visited = HashSet::new();

        loop {
            match parents.pop() {
                Some(node) => {
                    let id = node.id();
                    if visited.contains(&id) {
                        continue;
                    }

                    visited.insert(id);
                    node.backward_step();

                    for parent in node.backward_parents() {
                        if !visited.contains(&parent.id()) {
                            parents.push(parent);
                        }
                    }
                }
                None => break,
            }
        }
    }
}

impl<T: std::fmt::Debug> RecordedOpsParent for Node<T> {
    fn backward_step(&self) {
        println!("backward node {}", self.id);
        self.ops.backward_step(&self.state)
    }
    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        self.ops.backward_parents()
    }

    fn id(&self) -> String {
        self.id.clone()
    }
}

pub type NodeRef<Out> = Rc<Node<Out>>;

pub trait Zeros<T> {
    fn zeros(&self) -> T;
}
pub trait Ones<T> {
    fn ones(&self) -> T;
}
