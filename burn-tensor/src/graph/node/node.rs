use super::NodeStateRef;
use crate::{ops::RecordedOpsRef, tape::Tape};
use std::{collections::HashSet, rc::Rc};

#[derive(Debug)]
pub struct Node<Out> {
    pub id: usize,
    pub state: NodeStateRef<Out>,
    pub ops: RecordedOpsRef,
}

impl<Out> Node<Out> {
    pub fn new(state: NodeStateRef<Out>, ops: RecordedOpsRef) -> Self {
        let id = 0;
        println!("Creating new node with id {}", id);

        Self { id, state, ops }
    }
    pub fn from_binary<Lhs, Rhs>(
        lhs: &Node<Lhs>,
        rhs: &Node<Rhs>,
        state: NodeStateRef<Out>,
        ops: RecordedOpsRef,
    ) -> Self {
        let id = usize::max(lhs.id, rhs.id) + 1;
        println!("Creating new node with id {}", id);

        Self { id, state, ops }
    }
    pub fn from_single<Lhs>(
        input: &Node<Lhs>,
        state: NodeStateRef<Out>,
        ops: RecordedOpsRef,
    ) -> Self {
        let id = input.id + 1;
        println!("Creating new node with id {}", id);

        Self { id, state, ops }
    }
}

impl<Out> Node<Out> {
    pub fn record(&self, tape: &mut Tape) {
        let mut visited = HashSet::new();
        let mut ops_queue = self.ops.parents_ops();

        self.ops.set_last_ops();
        tape.add(self.ops.clone());

        loop {
            if ops_queue.len() == 0 {
                break;
            }
            let ops = ops_queue.pop().unwrap();

            for neighbor in ops.ops.parents_ops() {
                if !visited.contains(&neighbor.id) {
                    ops_queue.push(neighbor);
                }
            }

            if !visited.contains(&ops.id) {
                visited.insert(ops.id);
                tape.add(ops.ops.clone());
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
