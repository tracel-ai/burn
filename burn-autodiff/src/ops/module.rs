use super::unary_ops_wrapper;
use crate::graph::converter::Forward2BackwardGraphConverter;
use crate::graph::node::{BackwardNode, BackwardNodeRef, BackwardNodeState, ForwardNodeRef};
use crate::graph::ops::{
    BackwardRecordedOps, BackwardRecordedOpsBoxed, ForwardRecordedOps, RecordedOpsParentRef,
    UnaryOps, UnaryOpsNodeState,
};
use crate::tensor::ADTensor;
use crate::ADBackendDecorator;
use burn_tensor::backend::Backend;
use burn_tensor::ops::*;
use std::sync::Arc;

#[derive(new, Debug)]
struct EmbeddingBackward<B: Backend> {
    indexes: <B::IntegerBackend as Backend>::TensorPrimitive<2>,
}

impl<B: Backend> UnaryOps<B::TensorPrimitive<2>, B::TensorPrimitive<3>> for EmbeddingBackward<B> {
    fn partial(
        &self,
        state: &UnaryOpsNodeState<B::TensorPrimitive<2>, B::TensorPrimitive<3>>,
    ) -> B::TensorPrimitive<2> {
        B::embedding_backward(&state.input.value, &state.output.grad(), &self.indexes)
    }
}

impl<B: Backend> ModuleOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn embedding(
        weights: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<2>,
        indexes: &<<ADBackendDecorator<B> as Backend>::IntegerBackend as Backend>::TensorPrimitive<
            2,
        >,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<3> {
        let input = weights.node.clone();
        let output = B::embedding(weights.tensor_ref(), indexes);
        let ops = EmbeddingBackward::<B>::new(indexes.clone());

        unary_ops_wrapper(input, output, ops)
    }

    fn embedding_backward(
        weights: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<2>,
        output: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<3>,
        indexes: &<<ADBackendDecorator<B> as Backend>::IntegerBackend as Backend>::TensorPrimitive<
            2,
        >,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<2> {
        let tensor = B::embedding_backward(weights.tensor_ref(), output.tensor_ref(), indexes);
        ADTensor::from_tensor(tensor)
    }

    fn conv2d(
        x: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<4>,
        weight: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<4>,
        bias: &Option<<ADBackendDecorator<B> as Backend>::TensorPrimitive<1>>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<4> {
        #[derive(new, Debug)]
        pub struct ForwardConv2dOps<B: Backend> {
            x: ForwardNodeRef<B::TensorPrimitive<4>>,
            weights: ForwardNodeRef<B::TensorPrimitive<4>>,
            bias: Option<ForwardNodeRef<B::TensorPrimitive<1>>>,
        }

        #[derive(new, Debug)]
        pub struct BackwardConv2dOps<B: Backend> {
            x: BackwardNodeRef<B::TensorPrimitive<4>>,
            weights: BackwardNodeRef<B::TensorPrimitive<4>>,
            bias: Option<BackwardNodeRef<B::TensorPrimitive<1>>>,
        }

        impl<B: Backend> ForwardRecordedOps<B::TensorPrimitive<4>> for ForwardConv2dOps<B> {
            fn to_backward(
                &self,
                graph: &mut Forward2BackwardGraphConverter,
            ) -> BackwardRecordedOpsBoxed<B::TensorPrimitive<4>> {
                let bias = match &self.bias {
                    Some(bias) => Some(Arc::new(BackwardNode::from_node(bias, graph))),
                    None => None,
                };
                let ops = BackwardConv2dOps::<B>::new(
                    Arc::new(BackwardNode::from_node(&self.x, graph)),
                    Arc::new(BackwardNode::from_node(&self.weights, graph)),
                    bias,
                );

                Box::new(ops)
            }
        }

        impl<B: Backend> BackwardRecordedOps<B::TensorPrimitive<4>> for BackwardConv2dOps<B> {
            fn backward_step(&self, state: &BackwardNodeState<B::TensorPrimitive<4>>) {
                self.weights.state.update_grad(todo!());
                self.x.state.update_grad(todo!());

                if let Some(bias) = &mut self.bias {
                    bias.state.update_grad(todo!());
                }
            }

            fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
                let mut parents: Vec<RecordedOpsParentRef> = Vec::with_capacity(3);

                parents.push(self.x.clone());
                parents.push(self.weights.clone());

                if let Some(bias) = &self.bias {
                    parents.push(bias.clone());
                }

                parents
            }
        }

        let out = B::conv2d(
            x.tensor_ref(),
            weight.tensor_ref(),
            &match bias {
                Some(b) => Some(b.tensor()),
                None => None,
            },
            stride,
            padding,
        );

        let shape = *B::shape(&out);
        let mut order = usize::max(weight.node.order, x.node.order);
        if let Some(bias) = bias {
            order = usize::max(order, bias.node.order);
        }
        let state = crate::graph::node::ForwardNodeState::new(out);

        let ops = ForwardConv2dOps::<B>::new(
            x.node.clone(),
            weight.node.clone(),
            match bias {
                Some(b) => Some(b.node.clone()),
                None => None,
            },
        );
        let ops = Box::new(ops);

        let node = crate::graph::node::ForwardNode::new(order, state, ops);
        let node = std::sync::Arc::new(node);

        ADTensor { node, shape }
    }
}
