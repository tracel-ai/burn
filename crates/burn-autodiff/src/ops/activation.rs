use std::marker::PhantomData;

use crate::{
    checkpoint::{retro_forward::RetroForward, state::BackwardStates},
    grads::Gradients,
    graph::NodeID,
    ops::{unary, Backward, Ops, OpsKind},
    retro_unary, Autodiff, Checkpointer,
};
use burn_tensor::{
    backend::Backend,
    ops::{ActivationOps, FloatTensor},
};

impl<B: Backend> ActivationOps<Autodiff<B>> for Autodiff<B> {
    fn gelu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        #[derive(Debug)]
        struct Gelu<const D: usize>;

        retro_unary!(RetroGelu, B::gelu);

        impl<const D: usize, B: Backend> Backward<B, D, 1> for Gelu<D> {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);

                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::gelu_backward(input, grad)
                });
            }
        }

        match Gelu::<D>
            .prepare([tensor.node.clone()], [tensor.graph.clone()])
            .memory_bound()
            .retro_forward(RetroGelu::<B, D>::new(tensor.node.id.clone()))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::gelu(tensor.primitive.clone()))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::gelu(tensor.primitive)),
        }
    }

    fn relu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        #[derive(Debug)]
        struct Relu;

        retro_unary!(RetroRelu, B::relu);

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Relu {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let state = checkpointer.retrieve_node_output(ops.state);
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::relu_backward(state, grad)
                });
            }
        }

        match Relu
            .prepare([tensor.node.clone()], [tensor.graph.clone()])
            .memory_bound()
            .retro_forward(RetroRelu::<B, D>::new(tensor.node.id.clone()))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::relu(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::relu(tensor.primitive)),
        }
    }

    fn sigmoid<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        #[derive(Debug)]
        struct Sigmoid;

        retro_unary!(RetroSigmoid, B::sigmoid);

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Sigmoid {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                let output = B::sigmoid(input);
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::sigmoid_backward(output, grad)
                });
            }
        }

        match Sigmoid
            .prepare([tensor.node.clone()], [tensor.graph.clone()])
            .memory_bound()
            .retro_forward(RetroSigmoid::<B, D>::new(tensor.node.id.clone()))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::sigmoid(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::sigmoid(tensor.primitive)),
        }
    }
}
