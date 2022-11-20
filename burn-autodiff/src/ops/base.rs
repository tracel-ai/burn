use crate::graph::{
    node::{ForwardNode, ForwardNodeRef, ForwardNodeState},
    ops::{BinaryOps, ForwardBinaryRecordedOps, ForwardUnaryRecordedOps, UnaryOps},
};
use crate::tensor::ADTensor;
use burn_tensor::backend::Backend;
use std::sync::Arc;

pub fn unary_ops_wrapper_explicit<B1, B2, O, const D1: usize, const D2: usize>(
    input: ForwardNodeRef<B1::TensorPrimitive<D1>>,
    output: B2::TensorPrimitive<D2>,
    ops: O,
) -> ADTensor<D2, B2>
where
    B1: Backend,
    B2: Backend,
    O: UnaryOps<B1::TensorPrimitive<D1>, B2::TensorPrimitive<D2>> + 'static,
{
    let shape = *B2::shape(&output);
    let state = ForwardNodeState::new(output);

    let ops = Arc::new(ops);
    let ops = ForwardUnaryRecordedOps::new(input.clone(), ops);
    let ops = Box::new(ops);

    let node = ForwardNode::from_unary(&input, state, ops);
    let node = Arc::new(node);

    ADTensor { node, shape }
}

pub fn unary_ops_wrapper<B, O, const D1: usize, const D2: usize>(
    input: ForwardNodeRef<B::TensorPrimitive<D1>>,
    output: B::TensorPrimitive<D2>,
    ops: O,
) -> ADTensor<D2, B>
where
    B: Backend,
    O: UnaryOps<B::TensorPrimitive<D1>, B::TensorPrimitive<D2>> + 'static,
{
    unary_ops_wrapper_explicit::<B, B, O, D1, D2>(input, output, ops)
}

pub fn binary_ops_wrapper<B, O, const D1: usize, const D2: usize, const D3: usize>(
    lhs: ForwardNodeRef<B::TensorPrimitive<D1>>,
    rhs: ForwardNodeRef<B::TensorPrimitive<D2>>,
    output: B::TensorPrimitive<D3>,
    ops: O,
) -> ADTensor<D3, B>
where
    B: Backend,
    O: BinaryOps<B::TensorPrimitive<D1>, B::TensorPrimitive<D2>, B::TensorPrimitive<D3>> + 'static,
{
    let shape = *B::shape(&output);
    let state = ForwardNodeState::new(output);

    let ops = Arc::new(ops);
    let ops = ForwardBinaryRecordedOps::new(lhs.clone(), rhs.clone(), ops);
    let ops = Box::new(ops);

    let node = ForwardNode::from_binary(&lhs, &rhs, state, ops);
    let node = Arc::new(node);

    ADTensor { node, shape }
}
