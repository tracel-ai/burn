use super::ADTensor;
use crate::graph::{
    converter::Forward2BackwardGraphConverter,
    grad::{AsNode, Gradients},
    node::{BackwardNode, ForwardNode},
};
use crate::tensor::backend::Backend;

impl<B: Backend, const D: usize> ADTensor<D, B> {
    pub fn backward(&self) -> Gradients {
        let mut converter = Forward2BackwardGraphConverter::new();
        let mut node = BackwardNode::from_node(&self.node, &mut converter);
        std::mem::drop(converter);

        node.backward()
    }
}

impl<B: Backend, const D: usize> AsNode<B::TensorPrimitive<D>> for ADTensor<D, B> {
    fn as_node(&self) -> &ForwardNode<B::TensorPrimitive<D>> {
        &self.node
    }
}
