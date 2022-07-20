use super::NodeStateRef;
use crate::ops::RecordedOpsRef;
use std::{ops::Add, rc::Rc};

#[derive(Debug)]
pub struct Node<Out> {
    pub id: usize,
    pub state: NodeStateRef<Out>,
    pub ops: RecordedOpsRef<Out>,
}

impl<Out> Node<Out> {
    pub fn new(state: NodeStateRef<Out>, ops: RecordedOpsRef<Out>) -> Self {
        let id = 0;
        println!("Creating new node with id {}", id);

        Self { id, state, ops }
    }
    pub fn from_binary<Lhs, Rhs>(
        lhs: &Node<Lhs>,
        rhs: &Node<Rhs>,
        state: NodeStateRef<Out>,
        ops: RecordedOpsRef<Out>,
    ) -> Self {
        let id = usize::max(lhs.id, rhs.id) + 1;
        println!("Creating new node with id {}", id);

        Self { id, state, ops }
    }
    pub fn from_single<Lhs>(
        input: &Node<Lhs>,
        state: NodeStateRef<Out>,
        ops: RecordedOpsRef<Out>,
    ) -> Self {
        let id = input.id + 1;
        println!("Creating new node with id {}", id);

        Self { id, state, ops }
    }
}

impl<Out> Node<Out>
where
    Out: Zeros<Out> + Ones<Out> + Clone + Add<Output = Out>,
    Out: std::fmt::Debug,
{
    pub fn record(&self) {
        let grad = self.state.borrow().value().ones();

        self.state.borrow_mut().update_grad(grad);
        self.ops.backward(&self.state);

        let mut nodes = self.ops.parents();
        loop {
            if let Some(node) = nodes.pop() {
                node.backward();

                for neighbor in node.parents() {
                    nodes.push(neighbor);
                }
            } else {
                break;
            }
        }
    }
}
pub type NodeRef<Out> = Rc<Node<Out>>;

pub trait Zeros<T> {
    fn zeros(&self) -> T;
}
pub trait Ones<T> {
    fn ones(&self) -> T;
}
