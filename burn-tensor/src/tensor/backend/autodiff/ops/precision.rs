use crate::backend::autodiff::ADBackendDecorator;
use crate::backend::Backend;
use crate::ops::TensorOpsPrecision;
use crate::{define_ops, execute_ops};
use crate::{
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    tensor::backend::autodiff::ADTensor,
};

define_ops!(
    name ADTensorToPrecisionOps
);
define_ops!(
    name ADTensorFromFullPrecisionOps
);

impl<B: Backend, const D: usize>
    UnaryOps<B::TensorPrimitive<D>, <B::FullPrecisionBackend as Backend>::TensorPrimitive<D>>
    for ADTensorToPrecisionOps<B, D>
{
    fn partial(
        &self,
        state: &UnaryOpsNodeState<
            B::TensorPrimitive<D>,
            <B::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
        >,
    ) -> B::TensorPrimitive<D> {
        let grad = state.output.grad();
        B::TensorPrimitive::from_full_precision(grad)
    }
}

impl<B: Backend, const D: usize>
    UnaryOps<<B::FullPrecisionBackend as Backend>::TensorPrimitive<D>, B::TensorPrimitive<D>>
    for ADTensorFromFullPrecisionOps<B, D>
{
    fn partial(
        &self,
        state: &UnaryOpsNodeState<
            <B::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
            B::TensorPrimitive<D>,
        >,
    ) -> <B::FullPrecisionBackend as Backend>::TensorPrimitive<D> {
        let grad = state.output.grad();
        grad.to_full_precision()
    }
}

impl<B: Backend, const D: usize> TensorOpsPrecision<ADBackendDecorator<B>, D>
    for <ADBackendDecorator<B> as Backend>::TensorPrimitive<D>
{
    fn to_full_precision(&self) -> ADTensor<D, <B as Backend>::FullPrecisionBackend> {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsPrecision::to_full_precision(&self.tensor()),
            ops ADTensorToPrecisionOps::<B, D>::new(),
        )
    }

    fn from_full_precision(
        tensor_full: ADTensor<D, <B as Backend>::FullPrecisionBackend>,
    ) -> ADTensor<D, B> {
        let tensor = <B as Backend>::TensorPrimitive::from_full_precision(tensor_full.tensor());
        let shape = *crate::tensor::ops::TensorOpsUtilities::shape(&tensor);
        let state = crate::graph::node::ForwardNodeState::new(tensor);

        let ops = std::sync::Arc::new(ADTensorFromFullPrecisionOps::<B, D>::new());
        let ops = crate::graph::ops::ForwardUnaryRecordedOps::new(tensor_full.node.clone(), ops);
        let ops = std::sync::Arc::new(ops);

        let node = crate::graph::node::ForwardNode::from_unary(&tensor_full.node, state, ops);
        let node = std::sync::Arc::new(node);

        crate::tensor::backend::autodiff::ADTensor { node, shape }
    }
}

// crate::register_tch!();
// crate::register_ndarray!();
