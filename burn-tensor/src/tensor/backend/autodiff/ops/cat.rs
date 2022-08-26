use crate::graph::converter::Forward2BackwardGraphConverter;
use crate::graph::node::{BackwardNode, BackwardNodeRef, BackwardNodeState, ForwardNodeRef};
use crate::graph::ops::{
    BackwardRecordedOps, BackwardRecordedOpsRef, ForwardRecordedOps, RecordedOpsParentRef,
};
use crate::tensor::backend::backend::Backend;
use crate::tensor::{backend::autodiff::ADTensor, ops::*};
use std::sync::Arc;

#[derive(new, Debug)]
pub struct ForwardCatOps<T> {
    nodes: Vec<ForwardNodeRef<T>>,
}

#[derive(new, Debug)]
pub struct BackwardCatOps<T> {
    nodes: Vec<BackwardNodeRef<T>>,
}

impl<T> ForwardRecordedOps<T> for ForwardCatOps<T>
where
    T: Clone + Zeros<T> + std::fmt::Debug + std::ops::Add<T, Output = T> + 'static + Send + Sync,
{
    fn to_backward(&self, graph: &mut Forward2BackwardGraphConverter) -> BackwardRecordedOpsRef<T> {
        Arc::new(BackwardCatOps::new(
            self.nodes
                .iter()
                .map(|node| {
                    let ops: BackwardNode<T> = BackwardNode::from_node(node, graph);
                    Arc::new(ops)
                })
                .collect(),
        ))
    }
}

impl<T> BackwardRecordedOps<T> for BackwardCatOps<T>
where
    T: Clone + Zeros<T> + std::fmt::Debug + std::ops::Add<T, Output = T> + 'static + Send + Sync,
{
    fn backward_step(&self, _state: &BackwardNodeState<T>) {}

    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        self.nodes
            .iter()
            .map(|node| {
                let ops: RecordedOpsParentRef = node.clone();
                ops
            })
            .collect()
    }
}

impl<B: Backend, const D: usize> TensorOpsCat<B::Elem, D> for ADTensor<D, B> {
    fn cat(tensors: Vec<&Self>, dim: usize) -> Self {
        let nodes: Vec<_> = tensors.iter().map(|t| t.node.clone()).collect();
        let order = nodes.iter().map(|node| node.order).max().unwrap();

        let tensors_inner: Vec<B::TensorPrimitive<D>> =
            tensors.into_iter().map(|a| a.tensor()).collect();
        let tensors_inner_ref: Vec<&B::TensorPrimitive<D>> = tensors_inner.iter().collect();

        let out = TensorOpsCat::cat(tensors_inner_ref, dim);

        let shape = out.shape().clone();
        let state = crate::graph::node::ForwardNodeState::new(out);

        let ops = ForwardCatOps::new(nodes);
        let ops = Arc::new(ops);

        let node = crate::graph::node::ForwardNode::new(order, state, ops);
        let node = std::sync::Arc::new(node);

        ADTensor { node, shape }
    }
}
