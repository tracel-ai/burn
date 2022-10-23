use crate::{
    backend::{autodiff::ADTensor, Backend},
    graph::{
        node::{ForwardNode, ForwardNodeRef, ForwardNodeState},
        ops::{ForwardUnaryRecordedOps, UnaryOps},
    },
};
use std::sync::Arc;

pub fn unary_ops_wrapper<B, O, const D1: usize, const D2: usize>(
    input: ForwardNodeRef<B::TensorPrimitive<D1>>,
    output: B::TensorPrimitive<D2>,
    ops: O,
) -> ADTensor<D2, B>
where
    B: Backend,
    O: UnaryOps<B::TensorPrimitive<D1>, B::TensorPrimitive<D2>> + 'static,
{
    let shape = *B::shape(&output);
    let state = ForwardNodeState::new(output);

    let ops = Arc::new(ops);
    let ops = ForwardUnaryRecordedOps::new(input.clone(), ops);
    let ops = Arc::new(ops);

    let node = ForwardNode::from_unary(&input, state, ops);
    let node = Arc::new(node);

    ADTensor { node, shape }
}
