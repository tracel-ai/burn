use crate::backend::Backend;
use crate::{define_ops, execute_ops};
use crate::{
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    tensor::backend::autodiff::ADTensor,
};
use crate::{ops::TensorOpsPrecision, Element};

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

macro_rules! define_impl {
    (
        $backend:ty,
        $backend_inner:ty,
        $element:ident
    ) => {
        impl<E: $element, const D: usize> TensorOpsPrecision<$backend, D>
            for <$backend as Backend>::TensorPrimitive<D>
        where
            E: Element,
        {
            fn to_full_precision(
                &self,
            ) -> ADTensor<D, <$backend_inner as Backend>::FullPrecisionBackend> {
                execute_ops!(
                    input self.node.clone(),
                    out TensorOpsPrecision::to_full_precision(&self.tensor()),
                    ops ADTensorToPrecisionOps::<$backend_inner, D>::new(),
                )
            }

            fn from_full_precision(
                tensor_full: ADTensor<D, <$backend_inner as Backend>::FullPrecisionBackend>,
            ) -> ADTensor<D, $backend_inner> {
                let tensor = <$backend_inner as Backend>::TensorPrimitive::from_full_precision(tensor_full.tensor());
                let shape = crate::tensor::ops::TensorOpsUtilities::shape(&tensor).clone();
                let state = crate::graph::node::ForwardNodeState::new(tensor);

                let ops = std::sync::Arc::new(ADTensorFromFullPrecisionOps::<$backend_inner, D>::new());
                let ops = crate::graph::ops::ForwardUnaryRecordedOps::new(tensor_full.node.clone(), ops.clone());
                let ops = std::sync::Arc::new(ops);

                let node = crate::graph::node::ForwardNode::from_unary(&tensor_full.node.clone(), state, ops);
                let node = std::sync::Arc::new(node);

                crate::tensor::backend::autodiff::ADTensor { node, shape }
            }
        }
    };
}

crate::register_tch!();
crate::register_ndarray!();
