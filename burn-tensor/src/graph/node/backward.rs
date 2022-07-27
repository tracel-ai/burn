use super::{BackwardNodeState, ForwardNodeRef, Ones, Zeros};
use crate::graph::{
    converter::Forward2BackwardGraphConverter,
    grad::Gradients,
    ops::{BackwardRecordedOpsRef, RecordedOpsParent, RecordedOpsParentRef},
    traversal::{BreadthFirstSearch, GraphTraversal},
};
use std::{ops::Add, sync::Arc};

#[derive(Debug)]
pub struct BackwardNode<Out> {
    pub id: String,
    pub order: usize,
    pub state: BackwardNodeState<Out>,
    pub ops: BackwardRecordedOpsRef<Out>,
}
pub type BackwardNodeRef<Out> = Arc<BackwardNode<Out>>;

impl<Out: Clone + Zeros<Out>> BackwardNode<Out> {
    pub fn from_node(
        node: &ForwardNodeRef<Out>,
        converter: &mut Forward2BackwardGraphConverter,
    ) -> Self {
        BackwardNode {
            id: node.id.clone(),
            order: node.order,
            state: BackwardNodeState::new(node.state.value()),
            ops: node.ops.to_backward(converter),
        }
    }
}

impl<Out> BackwardNode<Out>
where
    Out: Zeros<Out> + Ones<Out> + Clone + Add<Output = Out>,
    Out: std::fmt::Debug + 'static,
{
    pub fn backward(&mut self) -> Gradients {
        let grad = self.state.value().ones();
        self.state.update_grad(grad);
        self.ops.backward_step(&mut self.state);

        let traversal = BreadthFirstSearch::new(&self);
        let mut tape = vec![Vec::new(); self.order];

        traversal.traverse(|node| {
            let order = node.order();
            if order == 0 {
                return;
            }
            match tape.get_mut(order) {
                Some(nodes) => nodes.push(node),
                None => {}
            };
        });

        for i in (1..self.order).rev() {
            if let Some(nodes) = tape.get(i) {
                for node in nodes {
                    node.backward_step();
                }
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
        // println!("backward node id={} order={}", self.id, self.order);
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
