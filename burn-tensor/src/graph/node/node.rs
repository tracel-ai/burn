use super::NodeStateRef;
use crate::{ops::RecordedOpsRef, tape::Tape};
use std::{collections::HashSet, rc::Rc};

#[derive(Debug)]
pub struct Node<Out> {
    pub id: String,
    pub state: NodeStateRef<Out>,
    pub ops: RecordedOpsRef,
}

impl<Out> Node<Out> {
    pub fn new(state: NodeStateRef<Out>, ops: RecordedOpsRef) -> Self {
        Self {
            id: nanoid::nanoid!(),
            state,
            ops,
        }
    }
}

impl<Out> Node<Out> {
    pub fn record(&self, tape: &mut Tape) {
        let mut visited = HashSet::new();
        let mut all_ops = self.ops.parents_ops();

        tape.add(self.ops.clone());

        loop {
            if all_ops.len() == 0 {
                return;
            }
            let ops = all_ops.pop().unwrap();

            if !visited.contains(&ops.id) {
                tape.add(ops.ops.clone());
                visited.insert(ops.id);
            }

            for neighbor in ops.ops.parents_ops() {
                if !visited.contains(&neighbor.id) {
                    all_ops.push(neighbor);
                }
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
