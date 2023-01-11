use super::unary_ops_wrapper;
use crate::graph::converter::Forward2BackwardGraphConverter;
use crate::graph::node::{
    BackwardNode, BackwardNodeRef, BackwardNodeState, ForwardNode, ForwardNodeRef, ForwardNodeState,
};
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
        bias: Option<&<ADBackendDecorator<B> as Backend>::TensorPrimitive<1>>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<4> {
        let out = B::conv2d(
            x.tensor_ref(),
            weight.tensor_ref(),
            bias.map(|b| b.tensor_ref()),
            stride,
            padding,
        );
        let shape = *B::shape(&out);
        let mut order = usize::max(weight.node.order, x.node.order);
        if let Some(bias) = bias {
            order = usize::max(order, bias.node.order);
        }

        let ops = ForwardConv::<B, 2, 4>::new(
            x.node.clone(),
            weight.node.clone(),
            bias.map(|b| b.node.clone()),
            stride,
        );
        let ops = Box::new(ops);
        let state = ForwardNodeState::new(out);
        let node = ForwardNode::new(order, state, ops);
        let node = std::sync::Arc::new(node);

        ADTensor { node, shape }
    }
    fn conv1d(
        x: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<3>,
        weight: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<3>,
        bias: Option<&<ADBackendDecorator<B> as Backend>::TensorPrimitive<1>>,
        stride: usize,
        padding: usize,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<3> {
        let out = B::conv1d(
            x.tensor_ref(),
            weight.tensor_ref(),
            bias.map(|b| b.tensor_ref()),
            stride,
            padding,
        );
        let shape = *B::shape(&out);
        let mut order = usize::max(weight.node.order, x.node.order);
        if let Some(bias) = bias {
            order = usize::max(order, bias.node.order);
        }

        let ops = ForwardConv::<B, 1, 3>::new(
            x.node.clone(),
            weight.node.clone(),
            bias.map(|b| b.node.clone()),
            [stride],
        );
        let ops = Box::new(ops);
        let state = ForwardNodeState::new(out);
        let node = ForwardNode::new(order, state, ops);
        let node = std::sync::Arc::new(node);

        ADTensor { node, shape }
    }
}

#[derive(new, Debug)]
pub struct ForwardConv<B: Backend, const D: usize, const S: usize> {
    x: ForwardNodeRef<B::TensorPrimitive<S>>,
    weights: ForwardNodeRef<B::TensorPrimitive<S>>,
    bias: Option<ForwardNodeRef<B::TensorPrimitive<1>>>,
    stride: [usize; D],
}

#[derive(new, Debug)]
pub struct BackwardConv<B: Backend, const D: usize, const S: usize> {
    x: BackwardNodeRef<B::TensorPrimitive<S>>,
    weights: BackwardNodeRef<B::TensorPrimitive<S>>,
    bias: Option<BackwardNodeRef<B::TensorPrimitive<1>>>,
    stride: [usize; D],
}

impl<B: Backend> ForwardRecordedOps<B::TensorPrimitive<4>> for ForwardConv<B, 2, 4> {
    fn to_backward(
        &self,
        graph: &mut Forward2BackwardGraphConverter,
    ) -> BackwardRecordedOpsBoxed<B::TensorPrimitive<4>> {
        let bias = self
            .bias
            .as_ref()
            .map(|bias| Arc::new(BackwardNode::from_node(bias, graph)));
        let ops = BackwardConv::<B, 2, 4>::new(
            Arc::new(BackwardNode::from_node(&self.x, graph)),
            Arc::new(BackwardNode::from_node(&self.weights, graph)),
            bias,
            self.stride,
        );

        Box::new(ops)
    }
}

impl<B: Backend> BackwardRecordedOps<B::TensorPrimitive<4>> for BackwardConv<B, 2, 4> {
    fn backward_step(&self, state: &BackwardNodeState<B::TensorPrimitive<4>>) {
        let grads = B::conv2d_backward(
            &self.x.state.value,
            &self.weights.state.value,
            self.bias.as_ref().map(|b| &b.state.value),
            self.stride,
            &state.grad.borrow(),
        );

        self.weights.state.update_grad(grads.weights_grad);
        self.x.state.update_grad(grads.x_grad);

        if let Some(bias) = &self.bias {
            if let Some(bias_grad) = grads.bias_grad {
                bias.state.update_grad(bias_grad);
            }
        }
    }

    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        match &self.bias {
            Some(bias) => vec![self.x.clone(), self.weights.clone(), bias.clone()],
            None => vec![self.x.clone(), self.weights.clone()],
        }
    }
}

impl<B: Backend> ForwardRecordedOps<B::TensorPrimitive<3>> for ForwardConv<B, 1, 3> {
    fn to_backward(
        &self,
        graph: &mut Forward2BackwardGraphConverter,
    ) -> BackwardRecordedOpsBoxed<B::TensorPrimitive<3>> {
        let bias = self
            .bias
            .as_ref()
            .map(|bias| Arc::new(BackwardNode::from_node(bias, graph)));
        let ops = BackwardConv::<B, 1, 3>::new(
            Arc::new(BackwardNode::from_node(&self.x, graph)),
            Arc::new(BackwardNode::from_node(&self.weights, graph)),
            bias,
            self.stride,
        );

        Box::new(ops)
    }
}

impl<B: Backend> BackwardRecordedOps<B::TensorPrimitive<3>> for BackwardConv<B, 1, 3> {
    fn backward_step(&self, state: &BackwardNodeState<B::TensorPrimitive<3>>) {
        let grads = B::conv1d_backward(
            &self.x.state.value,
            &self.weights.state.value,
            self.bias.as_ref().map(|b| &b.state.value),
            self.stride[0],
            &state.grad.borrow(),
        );

        self.weights.state.update_grad(grads.weights_grad);
        self.x.state.update_grad(grads.x_grad);

        if let Some(bias) = &self.bias {
            if let Some(bias_grad) = grads.bias_grad {
                bias.state.update_grad(bias_grad);
            }
        }
    }

    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        match &self.bias {
            Some(bias) => vec![self.x.clone(), self.weights.clone(), bias.clone()],
            None => vec![self.x.clone(), self.weights.clone()],
        }
    }
}
